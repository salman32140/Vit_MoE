from model import Expert, MoE
from m_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from transformers import AutoModel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import class_name_mapping as cnm
import optuna_config as optuna


if __name__ == "__main__":
    
    class_name_mapping = cnm.class_name_mapping
    print("class length", len(class_name_mapping))
    
    np.random.seed(3117)
    torch.manual_seed(3117)
    device = "cuda"
    
    batch_size = 32
    expert_learning_rate = 1e-3    
    input_dim = 768
    hidden_dim = 128*4
    output_dim = len(class_name_mapping)
    criterion = nn.CrossEntropyLoss()

    #optuna hyper-parameters
    top_k = optuna.BEST_NUM_EXPERTS
    total_epochs = optuna.BEST_NUM_EPOCH
    entropy_weight = optuna.BEST_ENTROPY_VAL
    orthogonal_weight = optuna.BEST_ORTHOGONAL_VAL
    gating_learning_rate = optuna.BEST_GATING_VAL  
    usage_weight = optuna.BEST_USAGE_VAL

    logging_name = "MOE_REG_PV200_FD2"
    
    train_data = CustomDataset(
        data_root="../../../dataset/plantVillage_TSNE_200",
        split_type="train",
        limit=False,
        model_name="google/vit-base-patch16-224-in21k"
    )
    
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])
    

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    experts = []    
    optimizers = [] 
    for i in range(top_k):
        expert = Expert(input_dim, hidden_dim, output_dim).to(device)
        experts.append(expert)
        optimizer = Adam(expert.parameters(), lr=expert_learning_rate)
        optimizers.append(optimizer)
    
    model = MoE(arr_experts=experts, input_dim=input_dim,top_k=top_k).to(device)
    optimizer_gating = Adam(model.gate.parameters(), lr=gating_learning_rate)
    opt_BB = Adam(model.vit.vit.encoder.parameters(), lr=optuna.BEST_LEARNING_RATE_VAL)

    for p in model.vit.parameters():
        p.requires_grad = False
    
    for p in model.vit.vit.encoder.parameters():
        p.requires_grad = True

    writer = SummaryWriter(str('runs/' + logging_name))
    best_val_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(total_epochs):
        model.train()
        model.addNoise = True
        
        print("Training:")
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for batch in tqdm(train_loader):
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            out, gate_out, expert_out = model(images)
            true_classes = torch.argmax(labels, dim=1)
            
            pre_target_G_clf = expert_out[torch.arange(len(expert_out)), :, true_classes]
            target_G_clf = torch.argmax(pre_target_G_clf, dim=1)
            
            loss_G_clf = criterion(gate_out, target_G_clf)
            loss_class = criterion(out, true_classes)
            loss = loss_G_clf + loss_class
            
            # entropy_loss = -torch.sum(gate_out * torch.log(gate_out + 1e-8), dim=1).mean()
            # loss -= entropy_weight * entropy_loss
            
            # orthogonal_loss = 0
            # for i in range(len(model.experts)):
            #     for j in range(i + 1, len(model.experts)):
            #         weight_i = model.experts[i].layer1.weight
            #         weight_j = model.experts[j].layer1.weight
            #         orthogonal_loss += torch.norm(torch.mm(weight_i, weight_j.T), p='fro')
            # loss += orthogonal_weight * orthogonal_loss

            # avg_gate_out = torch.mean(gate_out, dim=0)
            # usage_loss = torch.sum((avg_gate_out - 1 / len(model.experts))**2)
            # loss += usage_weight * usage_loss

            for optimizer in optimizers:
                optimizer.zero_grad()
                
            optimizer_gating.zero_grad()
            opt_BB.zero_grad()
            loss.backward()
            
            for optimizer in optimizers:
                optimizer.step()
                
            optimizer_gating.step()  
            opt_BB.step()
            running_loss += loss.item()
            
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_classes.cpu().numpy())
        
        avg_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        print(f'Epoch [{epoch+1}/{total_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}')
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        torch.save(model, f"checkpoints/{logging_name}_{epoch+1}.pt")
        
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        correct = 0
        total = 0
        with torch.no_grad():
            print("Validating:")
            model.addNoise = False
            
            for batch in tqdm(val_loader):
                images = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                out, gate_out, expert_out = model(images)
                true_classes = torch.argmax(labels, dim=1)
                
                loss_class = criterion(out, true_classes)
                val_loss += loss_class.item()
                
                preds = torch.argmax(out, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(true_classes.cpu().numpy())
                
                total += labels.size(0)
                correct += (preds == true_classes).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')
        
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model, f"checkpoints/best_{logging_name}.pt")

    writer.close()
    print(f'Best Achieved Acc: {best_val_accuracy:.4f}, Epoch: {best_epoch}')
