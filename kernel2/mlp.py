from dependencies import *
from loaddata import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class mlpmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(11*300,10)
        self.fc2 = nn.Linear(45,10)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50,60)
        self.fc4 = nn.Linear(60,3)
    def forward(self,embeddings,positions):
        e1 = [self.dropout1(embedding) for embedding in embeddings]
        e1 = [e.flatten(1) for e in e1]
        x1 = torch.cat(tuple([self.fc1(e) for e in e1]),1)
        x2 = torch.cat(tuple([self.fc2(position) for position in positions]),1)
        x = F.relu(torch.cat((x1,x2),1))
        xnorm = self.bn1(x)
        hiddenlayer = F.relu(self.fc3(xnorm))
        outs = F.softmax(self.fc4(hiddenlayer),dim=1)
        return outs


mod = mlpmodel()


a = torch.tensor(p_emb_dev,dtype=torch.float)
b = torch.tensor(a_emb_dev,dtype=torch.float)
c = torch.tensor(b_emb_dev,dtype=torch.float)
d = torch.tensor(pa_pos_dev,dtype=torch.float)
e = torch.tensor(pb_pos_dev,dtype=torch.float)

outs = mod([a,b,c],[d,e])

