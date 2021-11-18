import os 
import sys 
import copy 
import math 
import time 
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

class TrainCocatedModel(object):
    def __init__(self, final_model, image_data_loaders, tabular_data_loaders, scaler, criterion, optimizer, schedular, num_epochs, save_check_point_pth, early_stop):
        """
        Params:
        ------
        final_model | Model(model1, model2) object
        image_data_loaders | dict
        tabular_data_loaders | dict
        scalar | sklearn.preprocessing object
        criterion | torch.nn object
        optimizer | torch.optim object
        schedular | torch.optim object
        num_epochs | int
        save_check_point_pth | str 
        early_stop | int
        """
        self.final_model = final_model
        self.image_data_loaders = image_data_loaders
        self.tabular_data_loaders = tabular_data_loaders
        self.criterion = criterion
        self.scaler = scaler
        self.optimizer = optimizer
        self.schedular = schedular
        self.num_epochs = num_epochs
        self.save_check_point_pth = save_check_point_pth
        self.early_stop = early_stop

        self.image_train_loader = self.image_data_loaders['train']
        self.image_valid_loader = self.image_data_loaders['valid']
        self.tabular_train_loader = self.tabular_data_loaders['train']
        self.tabular_valid_loader = self.tabular_data_loaders['valid']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.history = {
            'train_loss' : [],
            'train_r2' : [],
            'valid_loss' : [],
            'valid_r2' : []
            }
    
    def train_model(self):
        start_time = time.time()
        self.best_model_wts = copy.deepcopy(self.final_model.state_dict())
        lowest_loss = float('inf')
        best_r2 = 0.0
        stop_count = 0

        print('Training started ....')
        batch_count = next(iter(self.image_train_loader))[0].shape[0]
        for epoch in range(self.num_epochs):
            print()
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            epoch_train_r2 = 0.0 
            epoch_train_loss = 0.0
            epoch_valid_loss = 0.0 
            epoch_valid_r2 = 0.0 

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.final_model.train()
                else: 
                    self.final_model.eval() 
                
                running_train_loss = 0.0 
                running_valid_loss = 0.0
                running_train_r2 = 0.0 
                running_valid_r2 = 0.0 
                batches_till = 0.0 

                for ((img_data, label), (tab_data, _)) in zip(self.image_data_loaders[phase], self.tabular_data_loaders[phase]):
                    batches_till += 1
                    X_img = img_data.to(self.device)
                    X_tab = tab_data.to(self.device)
                    y = label.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.final_model(X_img, X_tab)
                        loss = self.criterion(output, y)
                        predictions = output.view(output.size(0)).tolist()
                        gdt = y.view(y.size(0)).tolist()
                        if self.scaler:
                            predictions = self.scaler.inverse_transform(predictions)
                            gdt = self.scaler.inverse_transform(gdt)
                        r2_score_here = r2_score(gdt, predictions)
                        sys.stdout.write("\r{}> {} {} {} {} {} {} {}".format(
                            '=' * int(batches_till / batch_count),
                            phase,
                            'after_batch: ', batches_till,
                            'loss : ', [loss.item()],
                            'r2_score : ', [r2_score_here]))

                        sys.stdout.flush()
                        time.sleep(0.5)
                        
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    if phase == 'train':
                        running_train_loss += loss.item() * batch_count
                        running_train_r2 += r2_score_here
                        self.history['train_loss'].append(running_train_loss / batches_till)
                        self.history['train_r2'].append(running_train_r2 / batches_till)
                    
                    if phase == 'valid':
                        running_valid_loss += loss.item() * batch_count
                        running_valid_r2 += r2_score_here
                        self.history['valid_loss'].append(running_valid_loss / batches_till)
                        self.history['valid_r2'].append(running_valid_r2 / batches_till)

                if phase == 'train':
                    self.schedular.step()
                
                if phase == 'train':
                    epoch_train_loss = running_train_loss / len(self.image_data_loaders[phase].dataset)
                    epoch_train_r2 = running_train_loss / (batches_till)
                    print()
                    print('{} LOSS: {:.4f} R2 SCORE: {:.4f}'.format(phase, epoch_train_loss, epoch_train_r2))
                
                if phase == 'valid':
                    epoch_valid_loss = running_valid_loss / len(self.image_data_loaders[phase].dataset)
                    epoch_valid_r2 = running_valid_loss / (batches_till)
                    print()
                    print('{} LOSS: {:.4f} R2 SCORE: {:.4f}'.format(phase, epoch_valid_loss, epoch_valid_r2))
                
                if phase == 'valid':
                    average_best_metrics = (best_r2 + lowest_loss) / 2
                    average_epoch_metrics = (epoch_valid_loss + epoch_valid_r2) / 2

                    if epoch > 0 and average_epoch_metrics > average_best_metrics:
                        best_r2 = max(epoch_valid_r2, best_r2)
                        lowest_loss = min(lowest_loss, epoch_valid_loss)
                        print('saving best weights of the model ....')
                        self.best_model_wts = copy.deepcopy(self.final_model.state_dict())
                    
                    if epoch > 0 and epoch_valid_loss > lowest_loss:
                        stop_count += 1
                    else: 
                        stop_count = 0
                    
                    print() 
                    if stop_count >= self.early_stop:
                        print("Early stopping ....")
                        break 
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best validation R2 Score : {:.4f}'.format(best_r2))
        print('Lowest validation Loss: {:4f}'.format(lowest_loss))

        self.final_model.load_state_dict(self.best_model_wts)
        return self.final_model
        