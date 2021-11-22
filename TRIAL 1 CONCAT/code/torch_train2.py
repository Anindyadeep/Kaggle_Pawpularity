import warnings
from numpy.lib.function_base import average 
from tqdm import tqdm 
import torch 
import torch.nn as nn
import torch.optim as optim 
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'


class Train(object):
    def __init__(self, loaders, model, criterion, optimizer, num_epochs):
        super(Train, self).__init__()
        self.loaders = loaders
        self.final_model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_image_data_loader = self.loaders['train']['image']
        self.valid_image_data_loader = self.loaders['valid']['image']

        self.train_tab_data_loader = self.loaders['train']['tabular']
        self.valid_tab_data_loader = self.loaders['valid']['tabular']

        self.history = {
            'train_loss' : [],
            }
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            batches_till = 0.0
            for ((img_data, label), (tab_data, _)) in zip(self.train_image_data_loader, self.train_tab_data_loader):
                batches_till += 1
                X_img = img_data.to(device)
                X_tab = tab_data.to(device)
                y = label.to(device)
                self.optimizer.zero_grad()
                output = self.final_model(X_img, X_tab)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self.history['train_loss'].append(running_loss/batches_till)
                if batches_till % 999 == 0:
                    print(f"epoch {epoch} LOSS : {running_loss / 1000}")
                    running_loss = 0.0
    


import sys

class Train(object):
    def __init__(self, loaders, model, criterion, optimizer, num_epochs):
        super(Train, self).__init__()
        self.loaders = loaders
        self.final_model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_data_loader = self.loaders['image']
        self.tab_data_loader = self.loaders['tabular']

        self.history = {
            'train_loss' : [],
            'valid_loss' : [],
            }
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            batches_till = 0.0
            for ((img_data, label), (tab_data, _)) in zip(self.image_data_loader, self.tab_data_loader):
                batches_till += 1
                X_img = img_data.to(device)
                X_tab = tab_data.to(device)
                y = label.to(device)
                self.optimizer.zero_grad()
                output = self.final_model(X_img, X_tab)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self.history['train_loss'].append(running_loss/batches_till)
                sys.stdout("\r{}> {} {} {} {} {} {}".format(
                    '=' * int(batches_till / 64),
                    'Epoch: ',
                    epoch,
                    'Batch: ',
                    batches_till,
                    'Train Loss: ',
                    {running_loss / batches_till}
                ))
            print()
        return self.final_model


    def _rmse_score(self, out, y, scaler):
        pass 

    def validate_model(self, scaler):
        running_loss = 0.0 
        batches_till = 0.0 
        with torch.no_grad():
            for ((img_data, label), (tab_data, _)) in zip(self.valid_image_data_loader, self.valid_tab_data_loader):
                batches_till += 1
                X_img = img_data.to(device)
                X_tab = tab_data.to(device)
                y = label.to(device)
                outputs = self.final_model(X_img, X_tab)
                loss = self.criterion(outputs, y)
                running_loss += loss.item()
                self.history['valid_loss'].append(running_loss)
                sys.stdout("\r{}> {} {} {} {}".format(
                    '=' * int(batches_till / 64),
                    'Batch: ',
                    batches_till,
                    'Valid Loss: ',
                    {running_loss / batches_till}
                ))
        
            print()
        print(f"RMSE LOSS: {running_loss}")








#######################################################




