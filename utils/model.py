import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Original utils
from utils.base import *


# SE-block
class SELayer(nn.Module):

    def __init__(self, 
                 channel: int, 
                 reduction: int=8
                 ) -> None:
        
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        kernel_weights = y
        y = self.fc(y).view(b, c, 1, 1)
        return kernel_weights, x * y.expand_as(x)


# CNNModel with SE-block
class CNNModel(nn.Module):

    def __init__(self, 
                 batch_size: int=32, 
                 dropout_rate: float=0.5
                 ) -> None:
        
        super(CNNModel, self).__init__()
        # hyper parameters
        self.dropout_rate = dropout_rate
        self.batch_size  = batch_size

        # layers
        self.conv = nn.Conv2d(1,16,(4,9))
        self.pool = nn.MaxPool2d((1,6),stride=6)
        self.se = SELayer(16)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,2)
        self.dropout = nn.Dropout(self.dropout_rate)

    #
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor
                ) -> torch.Tensor:
        
        # RNA feature extraction module
        x1 = F.relu(self.conv(x1))
        _, x1 = self.se(x1)
        x1 = self.pool(x1)
        x1 = x1.view(self.batch_size,1,512)
        # Compound feature extraction module
        x2 = F.relu(self.fc1(x2))
        # Interaction prediction module
        x = torch.cat([x1,x2], dim=2)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return x.view(self.batch_size,2)

#
def train_process(
        model: CNNModel, 
        dataloader: torch.utils.data.DataLoader, 
        device: str, 
        optimizer: torch.optim.Adam, 
        criterion: torch.nn.CrossEntropyLoss, 
        l1_alpha: float
        ) -> dict:
    
    loss_list = []
    auroc_list = []
    auprc_list = []

    model.train()
    for X1, X2, y in dataloader:

        X1, X2, y = X1.to(device), X2.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X1, X2)
        loss = criterion(outputs, y)
        # L1 regulation ------------------------------
        l1 = torch.tensor(0., requires_grad=True)
        for w in model.parameters(): l1 = l1 + torch.norm(w, 1)
        loss = loss + l1_alpha * l1
        #---------------------------------------------
        loss.backward()
        optimizer.step()

        pred = torch.transpose(outputs, 1, 0)
        pred_list = pred[1].detach().cpu().numpy().tolist()
        true_list = y.detach().cpu().numpy().tolist()

        loss_list.append(loss.item())
        auroc_list.append(AUROC(true_list, pred_list))
        auprc_list.append(AUPRC(true_list, pred_list))

    return {'Loss': np.mean(loss_list), 'AUROC': np.mean(auroc_list), 'AUPRC': np.mean(auprc_list)}


#
def valid_process(
        model: CNNModel, 
        dataloader: torch.utils.data.DataLoader, 
        device: str,
        criterion: torch.nn.CrossEntropyLoss
        ) -> dict:
    
    loss_list = []
    auroc_list = []
    auprc_list = []

    model.eval()
    with torch.no_grad():
        for X1, X2, y in dataloader:

            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            outputs = model(X1, X2)
            loss = criterion(outputs, y)

            pred = torch.transpose(outputs, 1, 0)
            pred_list = pred[1].detach().cpu().numpy().tolist()
            true_list = y.detach().cpu().numpy().tolist()

            loss_list.append(loss.item())
            auroc_list.append(AUROC(true_list, pred_list))
            auprc_list.append(AUPRC(true_list, pred_list))

    return {'Loss': np.mean(loss_list), 'AUROC': np.mean(auroc_list), 'AUPRC': np.mean(auprc_list)}


#
def test_process(
        model: CNNModel, 
        dataloader: torch.utils.data.DataLoader, 
        device: str
        ) -> dict:

    sm = nn.Softmax(dim=0)

    model.eval()
    with torch.no_grad():
        for X1, X2, y in dataloader:

            X1, X2, y = X1.to(device), X2.to(device), y.to(device)
            outputs = model(X1, X2)

            pred = torch.transpose(outputs, 1, 0)
            pred = sm(pred)
            pred_list = pred[1].detach().cpu().numpy().tolist()
            true_list = y.detach().cpu().numpy().tolist()

    return {'Predict': pred_list, 'AUROC': AUROC(true_list, pred_list), 'AUPRC': AUPRC(true_list, pred_list)}


#
def scanning_motif(
        sequence: str, 
        model: CNNModel,
        device: str,
        window_size: int=9
        ) -> dict:

    # prepare one-hot representation of RNA sequence
    oh = seq2onehot(sequence)
    oh_on_device = torch.from_numpy(oh).float().unsqueeze(0).unsqueeze(0).to(device)

    lines = {}

    # prepare tensors of CNNModel's kernels
    kernels = [kernel[0].to('cpu').detach().numpy().copy() for kernel in model.conv.weight]
    kernel_weights, _ = model.se(F.relu(model.conv(oh_on_device)))
    kernel_weights = kernel_weights.to('cpu').detach().numpy().copy()

    # scanning process for each kernel
    for i,(weight,kernel) in enumerate(zip(kernel_weights[0],kernels)):

        ip = -float('inf')
        ip_idx = 0

        j = 0

        # calculate inner product as similarity score on each site
        while j+window_size <= len(sequence):

            sub_seq = [o[j:j+window_size] for o in oh]

            tmp = calc_inner_product(kernel, sub_seq)

            if tmp > ip: 
                ip = tmp
                ip_idx = j

            j += 1
        
        # preparation for output
        flag = ['-']*len(sequence)
        for k in range(window_size):
            flag[ip_idx+k] = sequence[ip_idx+k]
        line = ''.join(flag) + f' Filter_{i+1:0=2} (w={weight:.5f}, ip={ip:.5f})'

        lines[weight] = line

    return lines
