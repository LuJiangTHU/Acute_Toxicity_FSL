import argparse
import importlib as imp
from utils import *
import time
from CorrelationLearning import CorrelationLearning
import torch
import random
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
                    help = 'config file with parameters of the experiment. It is assumed that'
                           'the config file is placed on the directory ./config/.')
parser.add_argument('--checkpoint', type=int, default=0,
                    help = 'checkpoint (epoch id) that will be loaded. If a negative value is '
                           'given then the latest existing checkpoint exsiting checkpoint is loaded.')
parser.add_argument('--num_workers', type=int, default=2,
                    help = 'number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--disp_step', type=int, default=500, help = 'display step during training')
args_opt = parser.parse_args()


exp_config_file = os.path.join('.','config',args_opt.config+'.py')

# load the configuration parameters of the experiment
print('Experiment begin:{0}'.format(exp_config_file))
config = imp.machinery.SourceFileLoader('', exp_config_file).load_module().config
config['disp_step'] = args_opt.disp_step

# seting the random seed
seed_list = config['seed']
assert (type(seed_list)==list)

start_time = time.time()

for i, seed in enumerate(seed_list):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    exp_directory = os.path.join('.','experiments', args_opt.config+'_m{}'.format(i))
    config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored

    print('From configure file {1} loading configure {0}'.format(args_opt.config, exp_config_file))
    print('The logs produced from experiments, snapshots and model files will saved into {0}'.format(exp_directory))

    feature_name = config['feature_name']
    print('This experiment select molecular feature {0}'.format(feature_name))

    # Set the train and test datasets, and their corresponding data loaders
    data_train_opt = config['data_train_opt']
    data_test_opt = config['data_test_opt']

    train_fold_file = config['train_fold_file']
    test_fold_file = config['test_fold_file']



    dataset_train = ToxicityDataset(phase='train',
                                    dataset_file=train_fold_file,
                                    feature_name=feature_name)
    dataset_test  = ToxicityDataset(phase='test',
                                    dataset_file=test_fold_file,
                                    feature_name=feature_name)

    Dataloader_train = ToxicityDataloader(
            dataset = dataset_train,
            batch_size = data_train_opt['batch_size'],
            num_workers = args_opt.num_workers,
            epoch_size = data_train_opt['epoch_size']
        )

    Dataloader_test = ToxicityDataloader(
            dataset = dataset_test,
            batch_size = data_test_opt['batch_size'],
            num_workers = args_opt.num_workers,
            epoch_size = data_test_opt['epoch_size']
        )



    Alg = CorrelationLearning(opt=config)

    if args_opt.cuda:
        Alg.load_to_gpu()



    Alg.solve(data_loader_train=Dataloader_train,
             data_loader_test =Dataloader_test)

end_time = time.time()

elaspsed_time = end_time - start_time

print(f'training time on all models：{elaspsed_time} seconds。')