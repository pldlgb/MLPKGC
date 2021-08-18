import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

class MLP_1(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(MLP_1, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W_1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.W_2 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d2)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.W_3 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1+d2, d2)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.hidden_dropout3 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        W_ent = torch.mm(e1, self.W_1.view(e1.size(1), -1))
        # W_rel = W_rel.view(-1, e1.size(1), e1.size(1))
        W_ent = self.hidden_dropout1(W_ent)  
        # e1 = e1.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_rel = torch.mm(r, self.W_2.view(r.size(1), -1))
        # W_rel = W_rel.view(-1, e1.size(1), e1.size(1))
        W_rel = self.hidden_dropout2(W_rel)

        x = torch.mm(torch.cat((W_ent,W_rel),1),self.W_3)
        # x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

class MLP_2(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(MLP_2, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W_1 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.W_2 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d2)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        self.W_3 = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1+d2, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.hidden_dropout3 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()
        self.active = nn.ReLU()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        W_ent = torch.mm(e1, self.W_1.view(e1.size(1), -1))
        # W_rel = W_rel.view(-1, e1.size(1), e1.size(1))
        W_ent = self.hidden_dropout1(W_ent)  
        # e1 = e1.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_rel = torch.mm(r, self.W_2.view(r.size(1), -1))
        # W_rel = W_rel.view(-1, e1.size(1), e1.size(1))
        W_rel = self.hidden_dropout2(W_rel)

        x = torch.mm(torch.cat((W_ent,W_rel),1),self.W_3)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = self.active(x)
        x = torch.mm(x, self.E.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred
