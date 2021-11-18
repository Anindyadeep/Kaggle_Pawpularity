import os 
from pathlib import Path
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


criterion = nn.MSELoss()
optimizer = optim.Adam(main_model.parameters(), lr=0.000001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

train_model_config = tt.TrainCocatedModel(
            final_model=main_model,
            image_data_loaders=image_data_loaders,
            tabular_data_loaders=tabular_data_loaders,
            scaler=scaler,
            criterion=criterion,
            optimizer=optimizer,
            schedular=scheduler,
            num_epochs=1,
            save_check_point_pth='/home/anindya/Documents/kaggle/Pawpularity/TRIAL 1 CONCAT/model',
            early_stop=2)

main_model = train_model_config.train_model()