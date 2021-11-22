import os 
import numpy as np 
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms 
from PIL import Image 
print("=> data.py imported successfully")

# image dataset

class ImageDataset(Dataset):
    def __init__(self, base_path, csv_path, df, train, image_label_col_dict, transform):
        super(ImageDataset, self).__init__()
        self.base_path = base_path
        self.csv_path = csv_path
        self.df = df
        self.train = train
        self.transform = transform
        if df is not None:
            self.data = self.df
        else:
            self.data = pd.read_csv(self.csv_path)
        self.image_col = image_label_col_dict['image']
        self.label_col = image_label_col_dict['label']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data[self.image_col][idx]
        full_path = str(os.path.join(self.base_path, image_name) + '.jpg')
        image = Image.open(full_path)
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.data[self.label_col][idx]
            label = torch.tensor(label, dtype=torch.float32).view(1,)
        if not self.train:
            return image
        return (image, label)


# tabular dataset

class TabularDataset(Dataset):
    def __init__(self, csv_path, df, train, feature_label_col_dict):
        super(TabularDataset, self).__init__()
        self.csv_path = csv_path
        self.df = df 
        self.train = train 
        self.feature_label_col_dict = feature_label_col_dict
        if df is not None:
            self.data = self.df 
        else:
            self.data = pd.read_csv(self.csv_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature_start = self.feature_label_col_dict['feature'][0]
        feature_end = self.feature_label_col_dict['feature'][1]
        label = self.feature_label_col_dict['label']

        feature_data = np.array(self.data.iloc[:, feature_start:feature_end])
        label_data = np.array(self.data.iloc[:, label:])
        features = torch.tensor(feature_data[idx], dtype=torch.float32) 
        label = torch.tensor(label_data[idx], dtype=torch.float32).view(1)
        if not self.train:
            return features 
        return (features, label)