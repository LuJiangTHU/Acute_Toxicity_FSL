config = {

    'seed':[20240625,20240624,20240201,20230630,1230630,20300630,20240622,202419,2028,20280101],
    'train_fold_file': 'train_fold_0.txt',
    'test_fold_file': 'test_fold_0.txt',

    'feature_name': 'Avalon',


    'max_num_epochs': 120,

    'data_train_opt': {
        'batch_size': 32,
        'epoch_size': 100
    },

    'data_test_opt': {
        'batch_size': 24025,
        'epoch_size': 1
    },

    'learners': {

        'CorrelationNet': {
            'def_file': './models/CorrelationNet.py',
            'pretrained': None,

            'opt': {
                'in_features_dnn': 1024,
                'in_features_gcn': 26,
                'out_features': [768, 512, 384, 64],
                'num_layers': 4,
                'num_task': 59,
                'Dropout_p': 0.1

            },

            'optim_params': {
                'optim_type': 'sgd',
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'nesterov': True,
                'LUT_lr': [(20, 0.001), (40, 0.0006), (50, 0.00012), (60, 0.000024),
                           (70, 0.000012), (80, 0.000001), (100, 0.0000001), (120, 0.00000001)]

            }

        }

    }
}