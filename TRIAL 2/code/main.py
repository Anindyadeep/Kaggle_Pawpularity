import os 
import pandas as pd 
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data.dataloader import DataLoader  
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

# extra modules
import data as td 
import model as tm 
import train as tt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = StandardScaler()
print(device)

BASE_DIR = Path(__file__).resolve().parent.parent
train_valid_csv_path = os.path.join(BASE_DIR, 'Data/train.csv')
train_valid_base_path = os.path.join(BASE_DIR, 'Data/train')

data = pd.read_csv(train_valid_csv_path)
data[['Pawpularity']] = scaler.fit_transform(data[['Pawpularity']])

train_csv_data = data.iloc[:7929, :]
valid_csv_data = data.iloc[7929:, :]
train_csv_data = train_csv_data.reset_index(drop=True)
valid_csv_data = valid_csv_data.reset_index(drop=True)

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


# train

image_train_dataset = td.ImageDataset(
    base_path=train_valid_base_path,
    csv_path=None,
    df = train_csv_data,
    train = True,
    image_label_col_dict=image_label_col_dict,
    transform=train_transform)

tabular_train_dataset = td.TabularDataset(
    csv_path=None,
    df = train_csv_data,
    train=True,
    feature_label_col_dict=feature_label_col_dict
)

# valid

image_valid_dataset = td.ImageDataset(
    base_path=train_valid_base_path,
    csv_path=None,
    df=valid_csv_data,
    train=True,
    image_label_col_dict=image_label_col_dict,
    transform=valid_transform)

tabular_valid_dataset = td.TabularDataset(
    csv_path=None,
    df=valid_csv_data,
    train=True,
    feature_label_col_dict=feature_label_col_dict
)

# image loaders

image_train_loader = DataLoader(
    dataset=image_train_dataset,
    batch_size=64,
)

image_valid_loader = DataLoader(
    dataset=image_valid_dataset,
    batch_size=64,
)

# tabular loaders

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

loaders = {
    'image'   : image_data_loaders,
    'tabular' : tabular_data_loaders
}


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

image_model = tm.ViTModel().to(device)
tabular_model = tm.TabularModel().to(device)
main_model = tm.ViTTabularModel(image_model, tabular_model).to(device)


criterion = RMSELoss()
optimizer = optim.Adam(main_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

train_model_config = tt.Train(
    loaders = loaders,
    model = main_model,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = scheduler,
    num_epochs = 5,
    scaler = scaler
)

main_model = train_model_config.train_model()