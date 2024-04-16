import utilites
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryLogisticRegression(nn.Module):
    '''
    A binary logistic regression model. Note that the sigmoid function
    is applied during forward. Thus the appropriate loss is BCELoss https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    NOT BCEWithLogitsLoss.
    '''
    def __init__(self, inDim : int):
        super().__init__()
        self.inDim = inDim
        self.model = nn.Linear(inDim, 1)
        self.normalizer = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.inDim)
        out = self.model(x)
        out = self.normalizer(out)
        return out
    

class NN(nn.Module):
    def __init__(self,inDim : int):
        super(NN, self).__init__()
        self.inDim = inDim
        self.fc1 = nn.Linear(inDim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.normalizer = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, self.inDim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        out = self.normalizer(x)
        return out
    
    
class AdultNN(nn.Module):

    def __init__(self, inDim):
        super().__init__()

        self.fc1 = nn.Linear(inDim, inDim)
        self.fc2 = nn.Linear(inDim, 3)
        self.fc3 = nn.Linear(3,1)
        self.normalizer = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.normalizer(x)
        return x