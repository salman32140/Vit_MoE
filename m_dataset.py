import random
import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
from collections import defaultdict, Counter

class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        self.data_path      = kwargs["data_root"]
        model_name          = kwargs["model_name"]
        self.split_type     = kwargs["split_type"]
        self.limit          = kwargs["limit"]
        self.class_names = ['tomato_septoria_leaf_spot', 'apple_cedar_apple_rust', 'cherry_healthy', 'corn_common_rust', 'tomato_leaf_mold', 'apple_apple_scab', 'tomato_spider_mites_two_spotted_spider_mite', 'pepper_bell_bacterial_spot', 'potato_late_blight', 'peach_healthy', 'raspberry_healthy', 'potato_early_blight', 'pepper_bell_healthy', 'corn_cercospora_leaf_spot_gray_leaf_spot', 'grape_healthy', 'corn_northern_leaf_blight', 'apple_healthy', 'tomato_tomato_mosaic_virus', 'blueberry_healthy', 'tomato_bacterial_spot', 'tomato_late_blight', 'tomato_tomato_yellow_leaf_curl_virus', 'tomato_healthy', 'soybean_healthy', 'strawberry_healthy', 'tomato_early_blight', 'squash_powdery_mildew', 'grape_black_rot']
        self.data           = []
        self.class_map      = {}
        
        for root, class_name, files in os.walk(self.data_path):
            for c in class_name:
                class_path = os.path.join(root,c)
                count = 0
                for f in os.listdir(class_path):
                    if self.split_type == "train" and self.limit:
                        if count < 200:
                            self.data.append(f)
                            self.class_map[str(f)] = c
                            count += 1
                    else:
                        self.data.append(f)
                        self.class_map[str(f)] = c         
                
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            
    def __len__(self):
        return len(self.data)
    
    def get_labels(self, class_name):
        one_hot = np.zeros(len(self.class_names))
        one_hot[0] = 1 if class_name == "tomato_septoria_leaf_spot" else 0
        one_hot[1] = 1 if class_name == "apple_cedar_apple_rust" else 0
        one_hot[2] = 1 if class_name == "cherry_healthy" else 0
        one_hot[3] = 1 if class_name == "corn_common_rust" else 0
        one_hot[4] = 1 if class_name == "tomato_leaf_mold" else 0
        one_hot[5] = 1 if class_name == "apple_apple_scab" else 0
        one_hot[6] = 1 if class_name == "tomato_spider_mites_two_spotted_spider_mite" else 0
        one_hot[7] = 1 if class_name == "pepper_bell_bacterial_spot" else 0
        one_hot[8] = 1 if class_name == "potato_late_blight" else 0
        one_hot[9] = 1 if class_name == "peach_healthy" else 0
        one_hot[10] = 1 if class_name == "raspberry_healthy" else 0
        one_hot[11] = 1 if class_name == "potato_early_blight" else 0
        one_hot[12] = 1 if class_name == "pepper_bell_healthy" else 0
        one_hot[13] = 1 if class_name == "corn_cercospora_leaf_spot_gray_leaf_spot" else 0
        one_hot[14] = 1 if class_name == "grape_healthy" else 0
        one_hot[15] = 1 if class_name == "corn_northern_leaf_blight" else 0
        one_hot[16] = 1 if class_name == "apple_healthy" else 0
        one_hot[17] = 1 if class_name == "tomato_tomato_mosaic_virus" else 0
        one_hot[18] = 1 if class_name == "blueberry_healthy" else 0
        one_hot[19] = 1 if class_name == "tomato_bacterial_spot" else 0
        one_hot[20] = 1 if class_name == "tomato_late_blight" else 0
        one_hot[21] = 1 if class_name == "tomato_tomato_yellow_leaf_curl_virus" else 0
        one_hot[22] = 1 if class_name == "tomato_healthy" else 0
        one_hot[23] = 1 if class_name == "soybean_healthy" else 0
        one_hot[24] = 1 if class_name == "strawberry_healthy" else 0
        one_hot[25] = 1 if class_name == "tomato_early_blight" else 0
        one_hot[26] = 1 if class_name == "squash_powdery_mildew" else 0
        one_hot[27] = 1 if class_name == "grape_black_rot" else 0
        
        return torch.tensor(one_hot, dtype=torch.float32)
    
    def __getitem__(self, index):

        img_name    = self.data[index]
        img_path    = os.path.join(self.data_path, self.class_map[img_name], img_name)
        try:
            img         = Image.open(img_path)
            img         = self.image_processor(img).pixel_values
        except Exception as e:
            print("Exception")    
        img         = np.array(img)
        cname       = self.class_map[img_name]
        label       = self.get_labels(cname)
        return {
            "pixel_values": torch.tensor(img, dtype=torch.float32).squeeze(),
            "cname" : cname,
            "labels": label,
            "img_name" : img_name,
            }