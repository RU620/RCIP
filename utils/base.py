import numpy as np
import pandas
import random
import itertools
# PyTorch 
import torch
# Sckit-Learn
from sklearn.metrics import roc_curve, auc, average_precision_score


def fix_seed(seed: int) -> None:

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def second2date(seconds: float) -> str:

    h = seconds // 3600
    m = (seconds - 3600*h) // 60
    s = seconds - 3600*h - 60*m

    return f'{h:0>2}:{m:0>2}:{s:0>2}'


def describe_dataset(dataset: pandas.core.frame.DataFrame) -> dict:

    dataset = dataset.copy()

    return {
        'num_int': len(dataset),
        'num_int_pos': len(dataset[dataset['Label']==1]),
        'num_int_neg': len(dataset[dataset['Label']==0]),
        'num_rna': len(set(dataset['Sequence'].values)),
        'num_sm': len(set(dataset['SMILES'].values))
    }


def make_combination(param_grid: dict, sampling: float=None):

    param_names = list(param_grid.keys())
    param_vals =  list(param_grid.values())
    param_comb = list(itertools.product(*param_vals))

    if sampling is not None:
        param_comb = random.sample(param_comb, sampling)

    return param_names, param_comb


def AUROC(true_list: list, pred_list: list) -> float:

    fpr, tpr, _ = roc_curve(true_list, pred_list)
    return auc(fpr, tpr)


def AUPRC(true_list: list, pred_list: list) -> float:

    return average_precision_score(true_list, pred_list)


def seq2onehot(sequence: str) -> np.ndarray:

    sequence = sequence.replace('T','U')
    oh = []

    for base in sequence:
        if   base=='A': oh.append([1,0,0,0])
        elif base=='U': oh.append([0,1,0,0])
        elif base=='G': oh.append([0,0,1,0])
        elif base=='C': oh.append([0,0,0,1])
    while len(oh)<200: oh.append([0,0,0,0])

    return np.array(oh, dtype=np.float32).T


def calc_inner_product(mat_1: list, mat_2: list) -> float:

    val = 0

    for row_1, row_2 in zip(mat_1, mat_2):

        for v1, v2 in zip(row_1, row_2):

            val += (v1*v2)

    return val



