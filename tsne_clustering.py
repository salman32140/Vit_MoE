import os
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torchvision import models, transforms

# Load and preprocess images
def load_images_from_folder(folder, target_size=(224, 224)):
    images = []
    image_files = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(img_path)
            img = img.resize(target_size)
            img = np.array(img)
            if img.shape == (224, 224, 3):
                images.append(img)
                image_files.append(filename)
    return np.array(images), image_files

def extract_features(images, device):
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg16.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-3])  # Remove the last three layers
    vgg16.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = []
    with torch.no_grad():
        for img in images:
            img = preprocess(img).unsqueeze(0).to(device)
            feature = vgg16(img)
            features.append(feature.cpu().numpy().flatten())

    features = np.array(features)
    print(f"Extracted features shape: {features.shape}")
    return features

def cluster_images(features, n_clusters=200):
    print(f"Clustering features of shape: {features.shape}")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tsne_features)
    return clusters

def print_representative_images(clusters, image_files,source_folder,destination_folder):
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)
        representative_index = cluster_indices[0][0]
        selected_image = image_files[representative_index]
        selected_image_path = os.path.join(source_folder, selected_image)
        destination_image_path = os.path.join(destination_folder, selected_image)
        os.makedirs(os.path.dirname(destination_image_path), exist_ok=True)
        Image.open(selected_image_path).save(destination_image_path)
        
        

def main(source_folder,destination_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, image_files = load_images_from_folder(source_folder)
    print(f"Loaded {len(images)} images")
    features = extract_features(images, device)
    clusters = cluster_images(features)
    print_representative_images(clusters, image_files,source_folder,destination_folder)

if __name__ == "__main__":
    directory = '../../../dataset/plantVillage/'
    for folder_name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder_name)):
            image_folder = directory+'/'+folder_name
            destination_folder = directory+"_TSNE_200/"+folder_name
            main(image_folder,destination_folder)
    print("Done")