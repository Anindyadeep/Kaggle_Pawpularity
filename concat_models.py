import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import vgg11


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__() 
        self.VGG11 = vgg11(pretrained=True)
        self.VGG11 = self.VGG11.features
        for params in self.VGG11.parameters():
            params.requires_grad = False 
        
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features = 25088, out_features = 2048)
        self.linear2 = nn.Linear(in_features = 2048, out_features = 2048)
        self.drop = nn.Dropout(p = 0.3)
        self.linear3 = nn.Linear(in_features = 2048, out_features = 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.VGG11(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear3(x)
        return x
    

class TabularModel(nn.Module):
    def __init__(self):
        super(TabularModel, self).__init__()
        self.linear1 = nn.Linear(in_features = 12, out_features = 32)
        self.linear2 = nn.Linear(in_features = 32, out_features = 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x) 
        x = self.linear2(x)
        return x 

class ImageTabularModel(nn.Module):
    def __init__(self, model1, model2):
        super(ImageTabularModel, self).__init__()
        self.model1 = model1 
        self.model2 = model2 
        self.linear1 = nn.Linear(576, 128)
        self.linear2 = nn.Linear(128, 32)
        self.regressor = nn.Linear(32, 1)
    
    def forward(self, x1, x2):
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.regressor(x)
        return torch.sigmoid(x)
        