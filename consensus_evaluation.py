'''
This script is used to evaluate the 5 cross-validation R2, RMSE using 10 trained models
'''

import argparse
import importlib as imp
import csv

import numpy as np

from utils import *
from CorrelationLearning import CorrelationLearning

df = pd.read_csv('./data/dataset.txt')
endpoint_name_list = list(df.columns.values[2:])

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of data loading workers')
parser.add_argument('--disp_step', type=int, default=200, help='display step during training')
args_opt = parser.parse_args()

num_fold = 5
times = 10

R2_5CV_list = []
R2_avg_5CV_list = []

RMSE_5CV_list = []
RMSE_avg_5CV_list = []

for fold in range(0, num_fold):

    exp_config_file = os.path.join('.', 'config', 'cfg_ToxACoL_4layer_f{0}.py'.format(fold))
    exp_directory = [os.path.join('.', 'experiments', 'cfg_ToxACoL_4layer_f{0}_m{1}'.format(fold, i)) for i in range(times)]

    pred = []
    for i in range(0, times):  # 10 kinds of different training models

        config = imp.machinery.SourceFileLoader('', exp_config_file).load_module().config
        config['disp_step'] = args_opt.disp_step
        config['exp_dir'] = exp_directory[i]

        feature_name = config['feature_name']
        print('This experiment select molecular feature {0}'.format(feature_name))

        # Set the train and test datasets, and their corresponding data loaders
        data_test_opt = config['data_test_opt']
        test_fold_file = config['test_fold_file']

        dataset_test = ToxicityDataset(phase='test',
                                       dataset_file=test_fold_file,
                                       feature_name=feature_name)

        Dataloader_test = ToxicityDataloader(
            dataset=dataset_test,
            batch_size=data_test_opt['batch_size'],
            num_workers=args_opt.num_workers,
            epoch_size=data_test_opt['epoch_size']
        )

        fea_tst = dataset_test[0:][0]
        tar_tst = dataset_test[0:][1]
        tar_mask_tst = dataset_test[0:][2]

        Alg = CorrelationLearning(opt=config)

        Alg.load_checkpoint(epoch='*', train=False, suffix='.best')

        ## for correlation net
        CorrelationNet = Alg.learners['CorrelationNet']
        CorrelationNet.eval()
        pred.append(CorrelationNet(fea_tst))

    avg_pred = torch.stack(pred).mean(0)

    # print(avg_pred.shape)
    # with open('./table_results/estimation_result_fold2.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     header = ['endpoint_{}'.format(i) for i in range(59)]
    #     writer.writerow(header)
    #     for i in range(len(avg_pred)):
    #         writer.writerow(avg_pred[i].data.tolist())

    RMSE, RMSE_avg = calculate_RMSE(pred=avg_pred, target=tar_tst, target_mask=tar_mask_tst)
    R2, R2_avg = calculate_R2(pred=avg_pred, target=tar_tst, target_mask=tar_mask_tst)

    RMSE_5CV_list.append(RMSE)
    RMSE_avg_5CV_list.append(RMSE_avg)

    R2_5CV_list.append(R2)
    R2_avg_5CV_list.append(R2_avg)


RMSE_5CV = torch.stack(RMSE_5CV_list).mean(0).data.tolist()
RMSE_5CV_std = torch.stack(RMSE_5CV_list).std(0).data.tolist()
RMSE_avg_5CV = torch.stack(RMSE_avg_5CV_list).mean().item()
RMSE_avg_5CV_std = torch.stack(RMSE_avg_5CV_list).std().item()

R2_5CV = torch.stack(R2_5CV_list).mean(0).data.tolist()
R2_5CV_std = torch.stack(R2_5CV_list).std(0).data.tolist()
R2_avg_5CV = torch.stack(R2_avg_5CV_list).mean().item()
R2_avg_5CV_std = torch.stack(R2_avg_5CV_list).std().item()

print('Avg_RMSE (std):', RMSE_avg_5CV, RMSE_avg_5CV_std)
print('Avg_R2 (std):', R2_avg_5CV, R2_avg_5CV_std)

train_fold_file = config['train_fold_file']
data_train_opt = config['data_train_opt']
dataset_train = ToxicityDataset(phase='train', dataset_file=train_fold_file)
tar_mask_trn = dataset_train[0:][2]

# num_measurements_trn = []
# num_measurements_tst = []
# for i in range(59):
#     num_measurements_trn.append(torch.nonzero(tar_mask_trn[:, i]).squeeze().shape[0])
#     num_measurements_tst.append(torch.nonzero(tar_mask_tst[:, i]).squeeze().shape[0])
#
# num_each_endpoint = np.array(num_measurements_trn) + np.array(num_measurements_tst)

# save the performance into .csv
with open('./table_results/ToxACoL.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['Endpoint', 'RMSE', 'RMSE_std', 'R2', 'R2_std']
    writer.writerow(header)
    for i, endpoint in enumerate(endpoint_name_list):
        writer.writerow([endpoint] + [RMSE_5CV[i], RMSE_5CV_std[i], R2_5CV[i], R2_5CV_std[i]])
    writer.writerow(['Avg.',  RMSE_avg_5CV, RMSE_avg_5CV_std, R2_avg_5CV, R2_avg_5CV_std])

print('The results have been saved to ./table_results/ToxACoL.csv')