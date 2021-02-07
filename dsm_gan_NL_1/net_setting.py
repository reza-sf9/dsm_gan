def intilize_net_setting():
    network_settings = {'dim_input': None,
                        'hidden_size': [50, 50],
                        'hidden_size_dsm': [20],
                        'hidden_size_gan_t': [5],
                        'hidden_size_gan_x': [5],
                        'use_bias': True,
                        'mlp_type': 2,  # (2: 1 Layaer, 3: 2 Layer)
                        'num_experts': 6,  # Number of distribution in last layer
                        'distribution': 'LogNormal',  # ('LogNormal','Weibull')
                        'num_class': None,
                        'num_feature': None,
                        'init': False,
                        'kernel_initializer': 'glorot_normal',
                        'kernel_regularizer': 0e-5,
                        'activity_regularizer': 0}

    network_settings_pretrain = {'dim_input': None,
                                 'hidden_size': [50, 50],
                                 'mlp_type': 2,  # (2: 1 Layaer, 3: 2 Layer)
                                 'num_experts': 1,  # Number of distribution in last layer
                                 'distribution': network_settings['distribution'],  # ('LogNormal','Weibull')
                                 'num_class': None,
                                 'num_feature': None,
                                 'init': False,
                                 'kernel_initializer': 'glorot_normal',
                                 'kernel_regularizer': 0e-5,
                                 'activity_regularizer': 0,
                                 'method_loss': 1,
                                 'ELBO': True,
                                 'alpha': 1,
                                 }

    training_settings = {'data_path': 'mimic_small.csv',
                         'test_size': 0.2,
                         'x_fold': 5,
                         'seed': 1,
                         'missing_rate': 0.6,  # options: 0.2, 0.4, 0.6 or 0.8 (only for synthetic)
                         'dataset': 'MIMIC_NL',  # options: synthetic, METABRIC, MIMIC, SEER, SEER_sample, MIMIC_NL,
                         # MIMIC_NL_sample (NL: None Logitiutinal)
                         'size_batch_train': 256,
                         'size_batch_test': 256,
                         'num_iter_pretrain': 1,
                         'num_iter_UC': 100,
                         'num_iter_C': 100,
                         'loss_method': [4, 2],  # 1- 9
                         'learning_rate': 1e-3,
                         'lambda': 1e-2,
                         'alpha': 1,
                         'increase_censoring': False,
                         'threshold': 1e-3,
                         'ELBO': True,
                         'test_size': 0.1,
                         'use_validation_set': False,
                         'pretrain': True
                         }

    return network_settings, training_settings, network_settings_pretrain