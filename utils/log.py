import os
import datetime
import pandas
# Original utils
from utils.base import *


class Logger():

    def __init__(self, 
                 result_file_path: str, 
                 config: dict
                 ) -> None:

        self.log_file = f'{result_file_path}/log.txt'
        self.config = config

        # if log file doesn't exist, is will be created for each session
        if not os.path.isfile(self.log_file):

            session_id = result_file_path.split('/')[-1]

            f = open(self.log_file, 'w')
            f.write('#'*(len(session_id)+37)+'\n')
            f.write(f'##### LOG FILE FOR SESSION ID: {session_id} #####\n')
            f.write('#'*(len(session_id)+37)+'\n')
            f.write('\n')
            f.close()


    # update log file
    def write(self, 
              mode: str, 
              dataset: pandas.core.frame.DataFrame, 
              result: dict, 
              dt_now: datetime.datetime, 
              process_time: str, 
              trained_weight_path: str=None, 
              name: str=None
              ) -> None:

        f = open(self.log_file, 'a')

        desc = describe_dataset(dataset)
        num_int = desc['num_int']
        num_int_pos = desc['num_int_pos']
        num_int_neg = desc['num_int_neg']
        num_rna = desc['num_rna']
        num_sm = desc['num_sm']

        if mode=='param_tuning':

            name = self.config['dataset_train']
            num_fold = self.config['pt_num_fold']
            num_epoch = self.config['pt_num_epoch']
            params_grid = self.config['params_grid']

            best_params = result['best_params']
            best_scores = result['best_scores']
            auroc_mean = best_scores['AUROC_mean']
            auroc_std = best_scores['AUROC_std']
            auprc_mean = best_scores['AUPRC_mean']
            auprc_std = best_scores['AUPRC_std']

            f.write(f'Hyper parameter tuning ({dt_now})\n\n')
            f.write(f'  [argument] number of fold : {num_fold}\n')
            f.write(f'             number of epoch: {num_epoch}\n')
            f.write(f'             parameter grid : ') 
            for key,values in params_grid.items(): f.write(f'{key}({values}) ')
            f.write('\n\n')
            f.write(f'  [dataset]  {name}\n')
            f.write(f'             interaction  : {num_int}(pos-{num_int_pos}, neg-{num_int_neg})\n')
            f.write(f'             number of RNA: {num_rna}\n')
            f.write(f'             number of SM : {num_sm}\n\n')
            f.write(f'  [result]   process time ({process_time})\n')
            f.write(f'             best patameters: ')
            for key,values in best_params.items(): f.write(f'{key}({values}) ')
            f.write(f'\n\n')
            f.write(f'             best AUROC: {auroc_mean:.5f}+/-{auroc_std:.5f}\n')
            f.write(f'             best AUPRC: {auprc_mean:.5f}+/-{auprc_std:.5f}\n\n')


        elif mode=='train':

            if name is None:
                name = self.config['dataset_train']
            num_epoch = self.config['train_num_epoch']
            best_params = self.config['best_params']
            train_result_save_file = f'{name}_train_result.csv'

            auroc = result['AUROC']
            auprc = result['AUPRC']

            f.write(f'Model training ({dt_now})\n')
            f.write(f'  [argument] number of epoch : {num_epoch}\n')
            f.write(f'             hyper parameters: ') 
            for key,values in best_params.items(): f.write(f'{key}({values}) ') 
            f.write('\n\n')
            f.write(f'  [dataset]  {name}\n')
            f.write(f'             interaction  : {num_int}(pos-{num_int_pos}, neg-{num_int_neg})\n')
            f.write(f'             number of RNA: {num_rna}\n')
            f.write(f'             number of SM : {num_sm}\n\n')
            f.write(f'  [result]   process time ({process_time})\n')
            f.write(f'             result saved as: {train_result_save_file}\n')
            f.write(f'             trained weight saved at: {trained_weight_path}\n')
            f.write(f'             AUROC: {auroc:.5f}\n')
            f.write(f'             AUPRC: {auprc:.5f}\n\n')
        

        elif mode=='cv':

            name = self.config['dataset_train']
            num_fold = self.config['cv_num_fold']
            num_epoch = self.config['cv_num_epoch']
            best_params = self.config['best_params']

            auroc_mean = result['Valid']['AUROC_mean']
            auroc_std = result['Valid']['AUROC_std']
            auprc_mean = result['Valid']['AUPRC_mean']
            auprc_std = result['Valid']['AUPRC_std']

            f.write(f'Model evaluation [cross validation] ({dt_now})\n')
            f.write(f'  [argument] number of fold  : {num_fold}\n')
            f.write(f'             number of epoch : {num_epoch}\n')
            f.write(f'             hyper parameters: ') 
            for key,values in best_params.items(): f.write(f'{key}({values}) ')
            f.write('\n\n')
            f.write(f'  [dataset]  {name}\n')
            f.write(f'             interaction  : {num_int}(pos-{num_int_pos}, neg-{num_int_neg})\n')
            f.write(f'             number of RNA: {num_rna}\n')
            f.write(f'             number of SM : {num_sm}\n\n')
            f.write(f'  [result]   process time ({process_time})\n')
            f.write(f'             AUROC: {auroc_mean:.5f}+/-{auroc_std:.5f}\n')
            f.write(f'             AUPRC: {auprc_mean:.5f}+/-{auprc_std:.5f}\n\n')
        

        elif mode=='test':

            if name is None:
                name = self.config['dataset_test']
            best_params = self.config['best_params']
            test_result_save_file = f'{name}_test_result.csv'

            auroc = result['AUROC']
            auprc = result['AUPRC']

            f.write(f'Model evaluation [test] ({dt_now})\n')
            f.write(f'  [argument] hyper parameters   : ') 
            for key,values in best_params.items(): f.write(f'{key}({values}) ') 
            f.write('\n')
            f.write(f'             trained weight path: {trained_weight_path}\n\n')
            f.write(f'  [dataset]  {name}\n')
            f.write(f'             interaction  : {num_int}(pos-{num_int_pos}, neg-{num_int_neg})\n')
            f.write(f'             number of RNA: {num_rna}\n')
            f.write(f'             number of SM : {num_sm}\n\n')
            f.write(f'  [result]   process time ({process_time})\n')
            f.write(f'             result saved as: {test_result_save_file}\n')
            f.write(f'             AUROC: {auroc:.5f}\n')
            f.write(f'             AUPRC: {auprc:.5f}\n\n')
        
        f.close()
    

    # update log file
    def write_motif(self, sequence, lines, dt_now, trained_weight_path):

        lines = sorted(
            lines.items(),
            reverse=True
            )

        f = open(self.log_file, 'a')

        f.write(f'Motif detection ({dt_now})\n')
        f.write(f'  [argument] trained weight path : {trained_weight_path}\n\n')
        f.write(f'  {sequence}\n')
        for line in lines: f.write(f'  {line[1]}\n')
        f.write('\n')

        f.close()
