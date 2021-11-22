import sys 
import copy
import math 
import torch 
from sklearn.metrics import mean_squared_error, r2_score
from vit_pytorch.vit import ViT
import warnings
warnings.filterwarnings("ignore")
print("=> train.py imported successfully")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Train(object):
    def __init__(self, loaders, model, criterion, optimizer, scheduler, num_epochs, scaler):
        super(Train, self).__init__()
        self.loaders = loaders
        self.final_model = model 
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_image_data_loader = self.loaders['image']['train']
        self.valid_image_data_loader = self.loaders['image']['valid']

        self.train_tab_data_loader = self.loaders['tabular']['train']
        self.valid_tab_data_loader = self.loaders['tabular']['valid']
        self.scaler = scaler
        self.history = {
            'train_loss' : [],
            'valid_loss' : [],
            }
        self.scheduler = scheduler
    
    def train_model(self):
        for epoch in range(self.num_epochs):
            if self.scheduler:
                self.scheduler.step(epoch)
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
                sys.stdout.write("\r{}> {} {} {} {} {} {}".format(
                    '=' * int(batches_till / 64),
                    'Epoch: ',
                    epoch,
                    'Batch: ',
                    batches_till,
                    'Train Loss: ',
                    {running_loss / batches_till}
                ))
            print()
            self.validate_model(self.scaler)
            self.final_model = copy.deepcopy(self.final_model)
        return self.final_model


    def _rmse_score(self, out, y, scaler):
      out = out.tolist()
      y = y.tolist()
      preds = scaler.inverse_transform(out)
      preds = preds.reshape(len(preds))
      y = scaler.inverse_transform(y)
      y = y.reshape(len(y))
      mse = mean_squared_error(y, preds)
      return math.sqrt(mse)
      

    def validate_model(self, scaler):
        running_loss = 0.0 
        batches_till = 0.0 
        rmse_score = 0.0

        with torch.no_grad():
            for ((img_data, label), (tab_data, _)) in zip(self.valid_image_data_loader, self.valid_tab_data_loader):
                batches_till += 1
                X_img = img_data.to(device)
                X_tab = tab_data.to(device)
                y = label.to(device)
                outputs = self.final_model(X_img, X_tab)
                loss = self.criterion(outputs, y)
                running_loss += loss.item()
                rmse_score += self._rmse_score(outputs, y, scaler)

                self.history['valid_loss'].append(running_loss)
                sys.stdout.write("\r{}> {} {} {} {} {} {}".format(
                    '=' * int(batches_till / 64),
                    'Batch: ',
                    batches_till,
                    'Valid Loss: ',
                    {running_loss / batches_till},
                    'RMSE score: ',
                    {rmse_score / batches_till}
                ))
            print(f"RMSE : {rmse_score / batches_till}")