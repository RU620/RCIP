import time
import numpy as np
import pandas as pd
# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
# Sckit-Learn
from sklearn.model_selection import KFold
# Original utils
from utils.base import *
from utils.chem import *
from utils.model import *


# it generates X+Y+Z dataset and three testsets (A, B, C)
def split_data_for_example(mode: str, data_path: str):

    dataset = pd.read_csv(f'{data_path}/dataset.csv')

    if mode in ['train','cv']:
        return dataset[~dataset['ID_S'].isin(['S000058','S000131'])]
    
    elif mode=='test':
        test_a = dataset[dataset['ID_S']=='S000131']
        test_b = dataset[dataset['ID_S']=='S000058']
        test_c = pd.read_csv(f'{data_path}/testset_c.csv')
        return [test_a, test_b, test_c]


#
def create_loader(
        dataset: pd.core.frame.DataFrame,
        batch_size: int,
        shuffle: bool=True,
        num_workers: int=2,
        drop_last: bool=True
        ) -> torch.utils.data.DataLoader:

    dataset = dataset.copy()

    # RNA input
    Sequences = dataset['Sequence'].values.tolist()
    RNA_input = [seq2onehot(seq) for seq in Sequences]
    X1_array = np.array([[RNA] for RNA in RNA_input])
    X1 = torch.from_numpy(X1_array).float()

    # SM input
    SMILES = dataset['SMILES'].values.tolist()
    SM_input = [get_ecfp_from_smiles(smi) for smi in SMILES]
    X2_array = np.array([[sm] for sm in SM_input])
    X2 = torch.from_numpy(X2_array).float()

    # ground truth label
    y = dataset['Label'].values
    y = torch.from_numpy(y).long()

    # Data loader
    Dataset = torch.utils.data.TensorDataset(X1, X2, y)
    Dataloader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return Dataloader


# 
class RCIP():

    def __init__(self, cuda_id: int):

        self.device = f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu'
        self.model = CNNModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())


    def param_tuning(
            self,
            dataset: pd.core.frame.DataFrame,
            num_fold: int,
            num_epoch: int,
            params_grid: dict
            ) -> dict:
        
        param_names, params_comb = make_combination(params_grid)

        cur_score = 0
        best_params = []

        for i,params in enumerate(params_comb):

            temp_params = dict(zip(param_names, params))
            print(f'Comb_{i+1}/{len(params_comb)}')

            result = self.cv(
                dataset,
                num_fold,
                num_epoch,
                temp_params
            )
            score = result['Valid']['AUPRC_mean']

            if score>cur_score:
                cur_score = score
                best_scores = result['Valid']
                best_params = dict(zip(param_names, params))

        return {'best_params': best_params, 'best_scores': best_scores}


    def train(
            self, 
            dataset: pd.core.frame.DataFrame,
            best_params: dict,
            num_epoch: int,
            trained_weight_file: str
            ) -> dict:
        
        self.model.batch_size = best_params['batch_size']
        self.model.dropout_rate = best_params['dropout_rate']
        
        dataloader = create_loader(dataset, batch_size=best_params['batch_size'])

        loss_list = []
        auroc_list = []
        auprc_list = []

        bar_interval = num_epoch/10

        start_time = time.time()
        print('Progress Bar: ', end='')

        for epoch in range(num_epoch):

            metrics = train_process(
                self.model, 
                dataloader, 
                self.device, 
                self.optimizer, 
                self.criterion, 
                best_params['l1_alpha']
            )
            loss_list.append(metrics['Loss'])
            auroc_list.append(metrics['AUROC'])
            auprc_list.append(metrics['AUPRC'])

            if (epoch+1)%bar_interval==0: print('|', end='')

        process_time = time.time() - start_time
        print(f' {second2date(process_time)} ({process_time/num_epoch:.3f} s/epoch)')

        torch.save(self.model.state_dict(), trained_weight_file)

        result = pd.DataFrame(data={
            'Loss':loss_list, 
            'AUROC':auroc_list, 
            'AUPRC':auprc_list
            })

        return {'Result': result, 'AUROC': metrics['AUROC'], 'AUPRC': metrics['AUPRC']}
        

    def cv(
            self,
            dataset: pd.core.frame.DataFrame,
            num_fold: int,
            num_epoch: int,
            best_params: dict
            ) -> dict:
        
        self.model.batch_size = best_params['batch_size']
        self.model.dropout_rate = best_params['dropout_rate']

        train_auroc_list = []
        train_auprc_list = []
        valid_auroc_list = []
        valid_auprc_list = []

        bar_interval = num_epoch/10

        kf = KFold(n_splits=num_fold, shuffle=True)

        for i,(train_index,valid_index) in enumerate(kf.split(dataset)):

            trainset = dataset.iloc[train_index].reset_index()
            trainloader = create_loader(trainset, batch_size=best_params['batch_size'])
            validset = dataset.iloc[valid_index].reset_index()
            validloader = create_loader(validset, batch_size=best_params['batch_size'])

            print(f'[Fold_{i+1}] ', end='')

            start_time = time.time()
            print('Progress Bar: ', end='')

            for epoch in range(num_epoch):

                train_metrics = train_process(
                    self.model, 
                    trainloader, 
                    self.device, 
                    self.optimizer, 
                    self.criterion, 
                    best_params['l1_alpha']
                )

                valid_metrics = valid_process(
                    self.model, 
                    validloader, 
                    self.device,
                    self.criterion
                )
                
                if (epoch+1)%bar_interval==0: print('|', end='')

            train_auroc_list.append(train_metrics['AUROC'])
            train_auprc_list.append(train_metrics['AUPRC'])
            valid_auroc_list.append(valid_metrics['AUROC'])
            valid_auprc_list.append(valid_metrics['AUPRC'])

            process_time = time.time() - start_time
            print(f' {second2date(process_time)} ({process_time/num_epoch:.3f} s/epoch)')

        train_result = {
            'AUROC_mean': np.mean(train_auroc_list),
            'AUROC_std' : np.std(train_auroc_list),
            'AUPRC_mean': np.mean(train_auprc_list),
            'AUPRC_std' : np.std(train_auprc_list)
            }
        valid_result = {
            'AUROC_mean': np.mean(valid_auroc_list),
            'AUROC_std' : np.std(valid_auroc_list),
            'AUPRC_mean': np.mean(valid_auprc_list),
            'AUPRC_std' : np.std(valid_auprc_list)
            }

        return {'Train': train_result, 'Valid': valid_result}
 

    def test(
            self,
            dataset: pd.core.frame.DataFrame,
            best_params: dict,
            trained_weight_file: str
            ) -> dict:
        
        self.model.batch_size = len(dataset)
        self.model.dropout_rate = best_params['dropout_rate']
        self.model.load_state_dict(torch.load(trained_weight_file))
        
        dataloader = create_loader(dataset, batch_size=len(dataset), shuffle=False)

        results = test_process(
            self.model, 
            dataloader, 
            self.device
        )

        dataset['Predict'] = results['Predict']

        return {'Result': dataset, 'AUROC': results['AUROC'], 'AUPRC': results['AUPRC']}


    def motif_detection(
            self,
            sequence: str,
            best_params: dict,
            trained_weight_file: str
            ) -> dict:
        
        self.model.batch_size = best_params['batch_size']
        self.model.dropout_rate = best_params['dropout_rate']
        self.model.load_state_dict(torch.load(trained_weight_file))

        lines = scanning_motif(
            sequence,
            self.model,
            self.device,
            window_size=9
            )
        
        return lines

