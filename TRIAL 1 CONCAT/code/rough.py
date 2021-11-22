import os 
from pathlib import Path
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data.dataloader import DataLoader  
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

# extra modules
import torch_data as td
import concat_models as cm
import torch_train as tt 

 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

BASE_DIR = Path(__file__).resolve().parent.parent
train_valid_csv_path = os.path.join(BASE_DIR, 'Data/train.csv')
train_valid_base_path = os.path.join(BASE_DIR, 'Data/train')
train_df, valid_df = td.TrainValidSplit(train_valid_csv_path, None, 0.9).train_valid_split_df(drop_first=True)

scaler = StandardScaler()
train_df[['Pawpularity']] = scaler.fit_transform(train_df[['Pawpularity']])
valid_df[['Pawpularity']] = scaler.fit_transform(valid_df[['Pawpularity']])

# some lol jinis
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

image_label_col_dict = {
    'image' : 'Id',
    'label' : 'Pawpularity'
}

feature_label_col_dict = {
    'feature' : (1, -1),
    'label'   : -1
}


image_train_dataset = td.ImageDataset(
    base_path=train_valid_base_path,
    csv_path=None,
    df = train_df,
    train = True,
    image_label_col_dict=image_label_col_dict,
    transform=train_transform)

tabular_train_dataset = td.TabularDataset(
    csv_path=None,
    df = train_df,
    train=True,
    feature_label_col_dict=feature_label_col_dict
)

image_valid_dataset = td.ImageDataset(
    base_path=train_valid_base_path,
    csv_path=None,
    df=valid_df,
    train=True,
    image_label_col_dict=image_label_col_dict,
    transform=valid_transform)

tabular_valid_dataset = td.TabularDataset(
    csv_path=None,
    df=valid_df,
    train=True,
    feature_label_col_dict=feature_label_col_dict
)

image_train_loader = DataLoader(
    dataset=image_train_dataset,
    batch_size=64,
)

image_valid_loader = DataLoader(
    dataset=image_valid_dataset,
    batch_size=64,
)

tabular_train_loader = DataLoader(
    dataset=tabular_train_dataset,
    batch_size=64
)

tabular_valid_loader = DataLoader(
    dataset=tabular_valid_dataset,
    batch_size=64
)


image_data_loaders = {
    'train' : image_train_loader,
    'valid' : image_valid_loader
}

tabular_data_loaders = {
    'train' : tabular_train_loader,
    'valid' : tabular_valid_loader
}


# omk


image_model = cm.ImageModel().to(device)
tabular_model = cm.TabularModel().to(device)
main_model = cm.ImageTabularModel(image_model, tabular_model).to(device)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss



X_img = next(iter(image_train_loader))[0].to(device)
X_tab = next(iter(tabular_train_loader))[0].to(device)
y1 = next(iter(image_train_loader))[1].to(device)
y2 = next(iter(tabular_train_loader))[1].to(device)

out = main_model(X_img, X_tab)

preds = out.squeeze(dim=1).tolist()
y1 = y1.squeeze(dim=1).tolist()

preds = scaler.inverse_transform(preds)
y1 = scaler.inverse_transform(y1)

from sklearn.metrics import mean_squared_error
import math 

print(math.sqrt(mean_squared_error(y1, preds)))
