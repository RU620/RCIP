import os
import yaml
import time
import datetime
import warnings
warnings.simplefilter('ignore')
# Original utils
from utils.RCIP import *
from utils.log import *


def main(mode, example):


    #########################################
    #####  Load configuration & Set up  #####
    #########################################

    # directories
    cwd = os.getcwd()
    data_path = f'{cwd}/data'
    result_path = f'{cwd}/result'
    ckpt_path = f'{cwd}/ckpt'

    # load config.yaml
    with open(f'{cwd}/config.yaml','r') as f:
        config = yaml.safe_load(f)

    # session (identifier for model)
    session_id = config['session_id']
    result_save_path = f'{result_path}/{session_id}'
    if not os.path.isdir(result_save_path): os.mkdir(result_save_path)

    # cuda (default value is 0)
    cuda_id = config['cuda_id']
    
    # random seed (default value is 1234, which is used in this research)
    seed = config['random_seed']
    fix_seed(seed)

    # trained weight (if not defined, it will be automatically difined)
    if config['trained_weight_file'] is not None:
        trained_weight_file = config['trained_weight_file']
    else:
        name = config['dataset_train']
        bs = config['best_params']['batch_size']
        dr = str(config['best_params']['dropout_rate'])
        la = str(config['best_params']['l1_alpha'])
        ep = config['train_num_epoch']
        trained_weight_file = f'weight_{session_id}_{name}_bs{bs}_dr{dr[2:]}_la{la[2:]}_ep{ep}'
    trained_weight_path = f'{ckpt_path}/{trained_weight_file}.pth'

    # open log file as instance of Logger object
    logger = Logger(result_save_path, config)


    ##########################################
    #####  Model training & evaluation   #####
    ##########################################

    # model instantiation
    model = RCIP(cuda_id)

    print(f'< Mode={mode}, Example={example} >')

    dt_now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    # hyper parameter tuning
    if mode=='param_tuning':

        name = config['dataset_train']
        dataset = pd.read_csv(f'{data_path}/{name}.csv')

        start_time = time.time()
        result = model.param_tuning(
            dataset,
            config['pt_num_fold'],
            config['pt_num_epoch'],
            config['params_grid']
        )
        process_time = second2date(time.time() - start_time)

        logger.write(mode, dataset, result, dt_now, process_time)


    # model training
    elif mode=='train':

        # you can use X+Y+Z dataset, which is the same as this research
        if example: 
            name = 'X+Y+Z'
            dataset = split_data_for_example(mode, data_path)
        else:
            name = config['dataset_train']
            dataset = pd.read_csv(f'{data_path}/{name}.csv')
        
        start_time = time.time()
        result = model.train(
            dataset,
            config['best_params'],
            config['train_num_epoch'],
            trained_weight_path
        )

        df = result['Result']
        df.to_csv(f'{result_save_path}/{name}_train_result.csv')
        process_time = second2date(time.time() - start_time)

        logger.write(mode, dataset, result, dt_now, process_time, trained_weight_path, name)


    # model evaluation (cross validation)
    elif mode=='cv':

        # you can use X+Y+Z dataset, which is the same as this research
        if example: 
            name = 'X+Y+Z'
            dataset = split_data_for_example(mode, data_path)
        else:
            name = config['dataset_train']
            dataset = pd.read_csv(f'{data_path}/{name}.csv')

        start_time = time.time()
        result = model.cv(
            dataset,
            config['cv_num_fold'],
            config['cv_num_epoch'],
            config['best_params']
        )

        process_time = second2date(time.time() - start_time)

        logger.write(mode, dataset, result, dt_now, process_time)


    # model evaluation (testset)
    elif mode=='test':

        # you can use three testsets, which are the same as this research
        if example: 
            testset_names = ['test_a', 'test_b', 'test_c']
            datasets = split_data_for_example(mode, data_path)
        else:
            testset_names = config['dataset_test']
            datasets = [pd.read_csv(f'{data_path}/{name}.csv') for name in testset_names]

        for name,dataset in zip(testset_names, datasets):

            start_time = time.time()
            result = model.test(
                dataset,
                config['best_params'],
                trained_weight_path
            )

            df = result['Result']
            df.to_csv(f'{result_save_path}/{name}_test_result.csv')
            process_time = second2date(time.time() - start_time)

            logger.write(mode, dataset, result, dt_now, process_time, trained_weight_path, name)


    # RNA sequence motif detection
    elif mode=='motif_detection':

        sequences = config['sequence']

        for sequence in sequences:

            lines = model.motif_detection(
                sequence,
                config['best_params'],
                trained_weight_path
            )

            logger.write_motif(sequence, lines, dt_now, trained_weight_path)

    print(f'The detail of this trial was recorded at {logger.log_file}')


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                        type=str,
                        choices=['param_tuning','train','cv','test','motif_detection'],
                        help=''
                        )
    parser.add_argument('--example', 
                        action='store_true',
                        help=''
                        )
    args = parser.parse_args()

    print('')
    print('*** RCIP started ***')
    print('')

    main(mode=args.mode, example=args.example)

    print('')
    print('*** finished ***')
    print('')