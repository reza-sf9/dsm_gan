import tensorflow as tf
import numpy as np
import dsm_tf2
import dsm_utilites_tf2
import os
import import_data as impt
import dsm_plot_fun, net_setting
from tensorflow.keras.optimizers import Adam

# solving problem of running the project simultanioiusly several times
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.keras.backend.set_floatx('float64')

# %% Parameters
output_path = './Results/'
model_path = './Model/'
dir_path = os.path.dirname(os.path.realpath(__file__))  # get the working path of the code

# call function that initilize the configuration of the model
network_settings, training_settings, network_settings_pretrain = net_setting.intilize_net_setting()
data_mode = training_settings['dataset']
training_settings['pretrain'] = True
training_settings['uncensored'] = True
training_settings['censored'] = True
training_settings['dataset'] = 'MIMIC_NL'    # options: synthetic, METABRIC, MIMIC, SEER, SEER_sample, MIMIC_NL, MIMIC_NL_sample (NL: None
# Logitiutinal)
training_settings['size_batch_train'] = 256
training_settings['size_batch_test'] = 256
training_settings['num_iter_pretrain'] = 2
training_settings['num_iter_UC'] = 2
training_settings['num_iter_C'] = 2
training_settings['loss_method'] = [1, 2] # the first element specifies the loss of UC Net, the second element: loss of Cen Net

network_settings_pretrain['method_loss'] = 1    # 1: article loss, 2: mse loss
np.random.seed(training_settings['seed'])
tf.random.set_seed(training_settings['seed'])

for imlp in [2]:  # [2,3]
    for ihidden in [50]:  # [50,100,200,300]
        for iexpert in [2]:  # [6,8,15,20,30]

            network_settings['hidden_size'] = [ihidden, 50]
            network_settings['mlp_type'] = imlp
            network_settings['num_experts'] = iexpert
            network_settings_pretrain['hidden_size'] = [ihidden, 50]

            # %% load dataset
            print('Loading Data ...')

            data, events, times = impt.import_dataset(data_mode)
            num_feature = data.shape[1]
            network_settings['dim_input'] = num_feature

            data = data.astype('float64')
            times = times.astype('float64')

            # %% prepair Data
            quantiles = np.round(np.quantile(times[events != 0], [0.25, .5, .75, .99]).tolist()).astype(int)

            # splitting data
            DATA_tr_batch, num_batch_train, DATA_te_batch, num_batch_test, DATA_tr, DATA_te = dsm_utilites_tf2.data_split_fun(
                training_settings, data, times, events)

            [tr_data, tr_times, tr_events] = DATA_tr
            [te_data, te_times, te_events] = DATA_te

            optimizer = Adam(lr=training_settings['learning_rate'], beta_1=0.9, beta_2=0.999)
            # %% Pre-train Model
            '''
			Pretraining Model: based on DSM model, we have a pretraining section 
			'''
            model_pretrain = dsm_tf2.deep_survival_machines_pretrain(network_settings_pretrain)
            if training_settings['pretrain'] :
                # train and test the model for Pre-Train
                model_pretrain = dsm_utilites_tf2.main_train_test_preTr(network_settings_pretrain, training_settings, model_pretrain, optimizer,
                                                                        DATA_tr_batch, num_batch_train, DATA_te_batch, num_batch_test,
                                                                        quantiles, DATA_te, DATA_tr)

                network_settings['init'] = [model_pretrain.shape, model_pretrain.scale]

            '''Training Model: in this part we train the model in 2 parts
            1 training Uncensored (UC): just using uncensored cost function 
            2 training Cnesored (C): using the variance of uncensored data function here'''
            # %% Train Model UC
            training_settings['model_state'] = 'UC'

            model_UC = dsm_tf2.deep_survival_machines(network_settings)
            model_UC.shape = model_pretrain.shape
            model_UC.scale = model_pretrain.scale

            # train and test the model for UC
            if training_settings['uncensored']:
                model_UC, score_list_UC, loss_train_list_UC, loss_test_list_UC = dsm_utilites_tf2.main_train_test_UC_C(
                    network_settings, training_settings, model_UC, optimizer, quantiles, DATA_tr_batch, DATA_te_batch, num_batch_test,
                    num_batch_train, DATA_tr, DATA_te, tr_data, tr_times, tr_events, model_pretrain)

                dsm_plot_fun.plt_save_2(training_settings, loss_train_list_UC, loss_test_list_UC, output_path, 'UC')

            # %% Train Model Cen
            training_settings['model_state'] = 'Cen'

            model_C = model_UC

            # train and test the model for Cen
            if training_settings['censored']:
                model_C, score_list_C, loss_train_list_C, loss_test_list_C = dsm_utilites_tf2.main_train_test_UC_C(
                    network_settings, training_settings, model_C, optimizer, quantiles, DATA_tr_batch,
                    DATA_te_batch, num_batch_test, num_batch_train, DATA_tr, DATA_te, tr_data, tr_times, tr_events, model_UC)

                # plot results
                dsm_plot_fun.plt_save_2(training_settings, loss_train_list_C, loss_test_list_C, output_path, 'Cen')

            model = model_C




            # # %%  Save Plot 1
            # loss_train_C = loss_train_list_C[0]
            # loss_test_C = loss_test_list_C[0]
            # dsm_plot_fun.plt_save_1(model, dir_path, score_list_C, loss_train_C, loss_test_C)


