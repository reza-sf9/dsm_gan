import tensorflow as tf
import numpy as np
import import_data as impt
from sksurv.metrics import brier_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_ipcw
import dsm_tf2


def predict_LogNormal_cdf(model, DATA_te, t_horizon, pretrain_state):
    '''calculate lognormal_cdf

        Inputs
        model: trained model
        DATA_te: passed test Data (featurs)
        t_horizon: a vector of times to calculate cdf
        pretrain_state: show we use pretrain model or not '''

    data_te, times_te, labels_te = DATA_te
    if not pretrain_state:
        shape, scale, logits , disc_x_t_real, disc_x_t_gen = model(data_te, times_te)
    elif pretrain_state:
        shape, scale, logits = model(data_te, pre_training=True)

    t = tf.squeeze(times_te)
    e = tf.squeeze(labels_te)
    

    logits = tf.nn.log_softmax(logits,axis=1)
    logits = tf.cast(logits,'float64')
    
    k_ = tf.cast(shape,'float64')
    b_ = tf.cast(scale,'float64')
    
    mu = k_
    sigma = b_
    
    t_horz = tf.cast(t_horizon,'float64')
    t_horz = tf.repeat(tf.expand_dims(t_horz,axis=0),t.shape[0],axis=0)    
    
    cdfs = []
    pdfs = []
    hazards = []
    
    for j in range(len(t_horizon)):
        
        t = t_horz[:, j:j+1]
        
        f = - sigma - 0.5*tf.cast(np.log(2*np.pi),'float64') - tf.math.divide( (tf.math.log(t) - mu)**2, 2.*tf.math.exp(2*sigma)   )
        s = tf.math.divide (tf.math.log(t) - mu, tf.math.exp(sigma)*tf.cast(np.sqrt(2),'float64') )
        s = 0.5  - 0.5*tf.math.erf(s)
        s = tf.math.log(s+1e-6)

        lcdfs = s+logits
        lpdfs = f+logits

        lcdfs = tf.math.reduce_logsumexp(lcdfs, axis=1)
        lpdfs = tf.math.reduce_logsumexp(lpdfs, axis=1)

        cdfs.append(lcdfs)
        pdfs.append(lpdfs)
        hazards.append(lpdfs-lcdfs)

    return cdfs


def predict_Weibull_cdf(model, DATA_te, t_horizon):
    '''predict cdf for weibul distribution

    Inputs
        model: trained model
        DATA_te: passed test Data (featurs)
        t_horizon: a vector of times to calculate cdf
    '''
    data_te, times_te, labels_te = DATA_te

    shape, scale, logits = model(data_te, pre_training=False)

    # shape = shape.detach().cpu().numpy()
    # scale = scale.detach().cpu().numpy()
    # logits = logits.detach().cpu().numpy()

    t = tf.squeeze(times_te)
    e = tf.squeeze(labels_te)

    logits = tf.nn.log_softmax(logits, axis=1)
    logits = tf.cast(logits, 'float64')

    k_ = tf.cast(shape, 'float64')
    b_ = tf.cast(scale, 'float64')

    t_horz = tf.cast(t_horizon, 'float64')
    t_horz = tf.repeat(tf.expand_dims(t_horz, axis=0), t.shape[0], axis=0)

    cdfs = []
    pdfs = []
    hazards = []

    for j in range(len(t_horizon)):
        t = t_horz[:, j:j + 1]

        s = - (tf.math.pow(tf.math.exp(b_) * t, tf.math.exp(k_)))  # log survival
        f = k_ + b_ + ((tf.math.exp(k_) - 1) * (b_ + tf.math.log(t))) - (tf.math.pow(tf.math.exp(b_) * t, tf.math.exp(k_)))

        lcdfs = s + logits
        lpdfs = f + logits

        lcdfs = tf.math.reduce_logsumexp(lcdfs, axis=1)
        lpdfs = tf.math.reduce_logsumexp(lpdfs, axis=1)

        cdfs.append(lcdfs)
        pdfs.append(lpdfs)
        hazards.append(lpdfs - lcdfs)

    return cdfs

def predict_cdf(model, DATA_te, t_horizon, pretrain_state):
    '''in this def based on used distribution, we choose desired function to finding cdf

    Inputs:
    model: trained model
    DATA_te: passed Data (featurs)
    t_horizon: a vector of times to calculate cdf
    pretrain_state: show we use pretrain model or not
    '''
    if model.dist == 'Weibull':
        return predict_Weibull_cdf(model, DATA_te, t_horizon)
    
    if model.dist == 'LogNormal':
        return predict_LogNormal_cdf(model, DATA_te, t_horizon, pretrain_state)

def compute_CI_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state, risk=0):
    '''Compute CI score based on CDF

    Inputs:
    model: trained model
    quantiles:
    DATA_te: passed test Data (featurs)
    DATA_tr: passed train Data (featurs)
    t_horizon: a vector of times to calculate cdf
    pretrain_state: show we use pretrain model or not
    '''
    cdf_preds = predict_cdf(model, DATA_te, quantiles, pretrain_state)
    cdf_preds = [cdf.numpy() for cdf in cdf_preds]
    
    _, t_valid, e_valid = DATA_te
    _, t_train, e_train = DATA_tr
    
    t_train = t_train.astype('float64')
    t_tvalid = t_valid.astype('float64')
    
    e_train = e_train.astype('bool')
    e_valid = e_valid.astype('bool')
    
    uncensored = np.where(e_valid == 1)[0]
    
    et1 =  np.array([(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', int)])
    et2 =  np.array([(e_valid[i], t_valid[i]) for i in range(len(e_valid))], dtype=[('e', bool), ('t', int)])


    if(cdf_preds[0].shape[0]>0 and cdf_preds[1].shape[0]>0 and cdf_preds[2].shape[0]>0 and cdf_preds[3].shape[0]>0):
        cdf_ci_25 = concordance_index_ipcw(et1, et2, -cdf_preds[0], tau=quantiles[0])
        cdf_ci_50 = concordance_index_ipcw(et1, et2, -cdf_preds[1], tau=quantiles[1])
        cdf_ci_75 = concordance_index_ipcw(et1, et2, -cdf_preds[2], tau=quantiles[2])
        cdf_ci_m  = concordance_index_ipcw(et1, et2, -cdf_preds[3], tau=quantiles[3])
    else:
        cdf_ci_25 = (0,0)
        cdf_ci_50 = (0,0)
        cdf_ci_75 = (0,0)
        cdf_ci_m  = (0,0)
    

    return cdf_ci_25[0],  cdf_ci_50[0],  cdf_ci_75[0], cdf_ci_m[0]

def compute_Brier_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state, risk=0):
    '''Compute Brrier score based on CDF

        Inputs:
        model: trained model
        quantiles:
        DATA_te: passed test Data (featurs)
        DATA_tr: passed train Data (featurs)
        t_horizon: a vector of times to calculate cdf
        pretrain_state: show we use pretrain model or not
        '''

    cdf_preds = predict_cdf(model, DATA_te, quantiles, pretrain_state)
    cdf_preds = [np.exp(cdf.numpy()) for cdf in cdf_preds]
    
    _, t_valid, e_valid = DATA_te
    _, t_train, e_train = DATA_tr
    
    e_train = e_train.astype('bool')
    e_valid = e_valid.astype('bool')
    
    et1 =  np.array([(e_train[i], t_train[i]) for i in range(len(e_train))], dtype=[('e', bool), ('t', int)])
    et2 =  np.array([(e_valid[i], t_valid[i]) for i in range(len(e_valid))], dtype=[('e', bool), ('t', int)])

    cdf_br_25 = brier_score(et1, et2, cdf_preds[0], quantiles[0])
    cdf_br_50 = brier_score(et1, et2, cdf_preds[1], quantiles[1])
    cdf_br_75 = brier_score(et1, et2, cdf_preds[2], quantiles[2])
    cdf_br_m  = brier_score(et1, et2, cdf_preds[3], quantiles[3])
    

    return np.squeeze(cdf_br_25[1]),  np.squeeze(cdf_br_50[1]), np.squeeze(cdf_br_75[1]) ,np.squeeze(cdf_br_m[1])

def print_results(iteration, iter, mean_CI, best_CI, mean_Brier, best_Brier, loss_test, loss_train):
    '''printing scores in console'''

    print('       ', f'{iter:.1f}', end='')
    print('       ', f'{mean_CI:.4f}', end='')
    print('    ', f'{best_CI:.4f}', end='')
    print('      ', f'{mean_Brier:.4f}', end='')
    print('        ', f'{best_Brier:.4f}', end='')
    print('     ', f'{loss_test:.4f}', end='')
    print('     ', f'{loss_train:.4f}')

def data_split_fun(training_settings, data, times, events):
    '''this function has been used to split data into train, test, validation (if needed)

    Inputs:
    training_settings:
    data: total feature
    times: total time-to-even
    events: total events
    '''
    (tr_data, te_data, tr_times, te_times, tr_events, te_events) = \
        train_test_split(data, times, events,
                         test_size=training_settings['test_size'],
                         random_state=training_settings['seed'])

    use_validation_set = training_settings['use_validation_set']
    if use_validation_set:
        (tr_data, va_data, tr_times, va_times, tr_events, va_events) = \
            train_test_split(tr_data, tr_times, tr_events,
                             training_settings=['test_size'],
                             random_state=training_settings['seed'])

    # train data
    DATA_tr = [tr_data, tr_times, tr_events]
    DATA_tr_batch, num_batch_train = impt.get_mini_batch(training_settings['size_batch_train'], DATA_tr, shuffling=False)

    # validation data
    if use_validation_set:
        DATA_va = [va_data, va_times, va_events]
        test_batch_input = [va_data, va_times, va_events]

    # test data
    DATA_te = [te_data, te_times, te_events]
    if not use_validation_set:
        test_batch_input = [te_data, te_times, te_events]

    DATA_te_batch, num_batch_test = impt.get_mini_batch(training_settings['size_batch_test'], test_batch_input, shuffling=False)

    return DATA_tr_batch, num_batch_train, DATA_te_batch, num_batch_test, DATA_tr, DATA_te

def train_model_preTr(network_settings_pretrain, model, optimizer, DATA_tr_batch):
    '''this function train PRE-TRAIN model

    Inputs:
    network_settings_pretrain: configuration of NN
    model: the initial value for pre-train model
    optimizer:
    DATA_tr_batch: all batches of train data
    '''

    loss_epoch = 0.
    # count = 0
    for batch in DATA_tr_batch:
        # count += 1
        # print(count)
        mb_data, mb_times, mb_events = batch
        with tf.GradientTape(persistent=True) as tape:
            shape, scale, logits = model(mb_data, pre_training=True)
            loss = dsm_tf2.pretrain_loss(network_settings_pretrain, shape, scale, logits, mb_times, mb_events)
        gradients = tape.gradient(loss, model.variables, unconnected_gradients='zero')
        optimizer.apply_gradients(zip(gradients, model.variables))

        loss_epoch += loss

    return model, loss_epoch

def test_model_preTr(network_settings_pretrain, model, DATA_te_batch):
    '''this function test PRE-TRAIN model

        Inputs:
        network_settings_pretrain:
        model: the trained model for pre-train model
        DATA_te_batch: all batches of test data
        '''

    loss_epoch_te = 0
    # count = 0
    for batch_te in DATA_te_batch:
        # count += 1
        # print(count)
        mb_data_te, mb_times_te, mb_events_te = batch_te
        shape, scale, logits = model(mb_data_te, pre_training=True)
        loss_te = dsm_tf2.pretrain_loss(network_settings_pretrain, shape, scale, logits, mb_times_te, mb_events_te)
        loss_epoch_te += loss_te

    return loss_epoch_te

def main_train_test_preTr(network_settings_pretrain, training_settings, model, optimizer, DATA_tr_batch, num_batch_train, DATA_te_batch,
                          num_batch_test, quantiles, DATA_te, DATA_tr):
    '''this function train, test and print in the console for PRE-TRAIN model

        Inputs:
        network_settings_pretrain: configuration of pre-train model
        training_settings: configuratoin of training setting
        model: the initial value for pre-train model
        optimizer:
        DATA_tr_batch: all batches of train data
        num_batch_train: number of batch of train data
        DATA_te_batch: all batches of test data
        num_batch_test: number of batch of test data
        quantiles:
        DATA_te: all of test data
        DATA_tr: all of train data
        '''

    loss_pretrain = []
    loss_pretrain_te = []

    best_CI, best_Brier = 0., 1.
    score_CI, score_Brier = [], []
    list_CI, list_Brier = [], []

    oldloss_pretrain = 99999
    patience_pretrain = 0


    for i in range(training_settings['num_iter_pretrain']):
        # call train model
        model, loss_epoch = train_model_preTr(
            network_settings_pretrain, model, optimizer, DATA_tr_batch)

        loss_epoch = loss_epoch / num_batch_train
        loss_pretrain.append(loss_epoch.numpy())

        # call test model
        loss_epoch_te = test_model_preTr(network_settings_pretrain, model, DATA_te_batch)
        loss_epoch_te = loss_epoch_te / num_batch_test
        loss_pretrain_te.append(loss_epoch_te.numpy())

        # reza
        # %% assess train-test model_UC
        _, time_test, label_test = DATA_te
        pretrain_state = True
        CI = compute_CI_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state)

        Brier = compute_Brier_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state, risk=0)
        mean_Brier = np.mean(Brier)
        mean_CI = np.mean(CI)

        if mean_CI > best_CI:
            best_CI = mean_CI
        if mean_Brier < best_Brier:
            best_Brier = mean_Brier
        score_CI.append(best_CI)
        list_CI.append(CI)
        list_Brier.append(Brier)
        score_Brier.append(best_Brier)

        # %% print in the console
        if 0 == i:
            print('============================================================================================')
            print('                          Pre-Training - optimizing   Loss')
            print('--------------------------------------------------------------------------------------------')
            print('| % iteration | current CI | best CI | current Brier | best Brier | loss test | loss train |')
        print_results(training_settings['num_iter_UC'], i, mean_CI, best_CI, mean_Brier, best_Brier, loss_pretrain_te[-1],
                      loss_pretrain[-1])
        if np.abs(loss_pretrain_te[-1] - oldloss_pretrain) < training_settings['threshold']:
            patience_pretrain += 1
            if patience_pretrain == 3:
                break

        patience_pretrain = loss_pretrain_te[-1]

    return model
        # end reza
        # print('Epoch', str(i), 'Loss Train: ', str(loss_epoch.numpy()), 'Loss Valid: ', str(loss_epoch_te.numpy()))
        #
        # oldloss_pretrain = loss_pretrain_te[-1]

def finding_var_prev_model(training_settings, model_prev, tr_data, tr_times):
    '''fincing the variance based on previous model:

    Inputs
    training_settings: confugration of traning setting
    model_prev: previous trained model
    tr_data: all train feature
    tr_times: all train time-to-event
    '''
    if training_settings['model_state'] == 'UC':
        shape_tot_prev, scale_tot_prev, logit_tot_prev = model_prev(tr_data, pre_training=True)
    elif training_settings['model_state'] == 'Cen':
        shape_tot_prev, scale_tot_prev, logit_tot_prev, disc_x_t_real, disc_x_t_fake = model_prev(tr_data, tr_times)

    mu = shape_tot_prev
    sigma_2 = tf.math.exp(2 * scale_tot_prev)
    gate = logit_tot_prev

    var_prev = (tf.math.exp(sigma_2) - 1) * np.exp(2 * mu + sigma_2)
    var_prev = var_prev * (gate ** 2)

    return var_prev

def train_model_UC_C(network_settings, training_settings, model, optimizer, DATA_tr_batch, num_batch_train, var_pr, e_tot, t_train):
    '''this function  TRAIN model for Uncensored and Censored Loss

        Inputs:
        network_settings: configuration of NN
        training_settings: configuration of training
        model: the initial value for training model (based on pretrain model)
        optimizer:
        DATA_tr_batch: all batches of train data
        num_batch_train: number of batches of train data
        var_pr: variance of all data based on previous model
        e_tot: event of all data
        t_train: time to event of all data
    '''

    loss_epoch = 0
    loss_epoch_C = 0.
    loss_epoch_UC = 0.
    loss_epoch_disc_uc_1 = 0

    for batch in DATA_tr_batch:
        mb_data, mb_times, mb_events = batch
        with tf.GradientTape() as tape:
            shape, scale, logit, disc_x_t_real, disc_x_t_fake = model(mb_data, mb_times)
            loss, LossUC, lossC, disc_loss_uc_1 = dsm_tf2.loss_total(network_settings, training_settings,
                                                                     shape, scale, logit, disc_x_t_real, disc_x_t_fake,
                                                                     mb_times, mb_events, var_pr,
                                                                     e_tot, t_train)
        gradients = tape.gradient(loss, model.variables, unconnected_gradients='zero')
        optimizer.apply_gradients(zip(gradients, model.variables))
        loss_epoch += loss
        loss_epoch_C += lossC
        loss_epoch_UC += LossUC
        loss_epoch_disc_uc_1 += disc_loss_uc_1

    # train - accumulative loss
    loss_epoch = loss_epoch / num_batch_train
    loss_epoch_C = loss_epoch_C / num_batch_train
    loss_epoch_UC = loss_epoch_UC / num_batch_train
    loss_epoch_disc_uc_1 = loss_epoch_disc_uc_1 / num_batch_train

    return model, loss_epoch, loss_epoch_C, loss_epoch_UC, loss_epoch_disc_uc_1

def test_model_UC_C(network_settings, training_settings, model, DATA_te_batch, num_batch_test, var_pr, e_tot, t_train):
    '''this function  TEST model for Uncensored and Censored Loss

            Inputs:
            network_settings: configuration of NN
            training_settings: configuration of training
            model: the trained value for training model
            optimizer:
            DATA_te_batch: all batches of test data
            num_batch_train: number of batches of test data
            var_pr: variance of all data based on previous model
            e_tot: event of all data
            t_train: time to event of all data
        '''
    loss_epoch_te = 0.
    loss_epoch_te_C = 0.
    loss_epoch_te_UC = 0.
    loss_epoch_te_disc_uc_1 = 0

    for batch_te in DATA_te_batch:
        mb_data_te, mb_times_te, mb_events_te = batch_te
        shape, scale, logits, disc_x_t_real, disc_x_t_fake = model(mb_data_te, mb_times_te, training=False)
        loss_te, lossC_te, LossUC_te, disc_loss_uc_1_te = dsm_tf2.loss_total(network_settings, training_settings,
                                                                             shape, scale, logits, disc_x_t_real, disc_x_t_fake,
                                                                             mb_times_te, mb_events_te,
                                                                             var_pr, e_tot, t_train)
        loss_epoch_te += loss_te
        loss_epoch_te_C += lossC_te
        loss_epoch_te_UC += LossUC_te
        loss_epoch_te_disc_uc_1 += disc_loss_uc_1_te

    # test - accumulative loss
    loss_epoch_te = loss_epoch_te / num_batch_test
    loss_epoch_te_C = loss_epoch_te_C / num_batch_test
    loss_epoch_te_UC = loss_epoch_te_UC / num_batch_test
    loss_epoch_te_disc_uc_1 = loss_epoch_te_disc_uc_1 / num_batch_test

    return loss_epoch_te, loss_epoch_te_C, loss_epoch_te_UC, loss_epoch_te_disc_uc_1

def main_train_test_UC_C(network_settings, training_settings, model, optimizer, quantiles, DATA_tr_batch, DATA_te_batch, num_batch_test,
                         num_batch_train, DATA_tr, DATA_te, tr_data, tr_times, tr_events, model_prev):
    '''this function  TRAIN model, TEST model, PRINT RESULT in console  for Uncensored and Censored Loss

            Inputs:
            network_settings: configuration of NN
            training_settings: configuration of training
            model: the initial value for training model (based on pretrain model)
            optimizer:
            quantiles:
            DATA_tr_batch: all batches of train data
            DATA_te_batch: all batches of test data
            num_batch_test: number of batches of test data
            num_batch_train: number of batches of train data
            DATA_tr: all train data
            DATA_te: all test data
            tr_events: event of all data
            tr_times: time to event of all data
            model_prev: previous model (used for finding variance)
        '''


    if training_settings['model_state'] == 'UC':
        num_iter = training_settings['num_iter_UC']

    elif training_settings['model_state'] == 'Cen':
        num_iter = training_settings['num_iter_C']


    # finding variance based preveious model
    var_prev = finding_var_prev_model(training_settings, model_prev, tr_data, tr_times)

    loss_train, loss_test = [], []

    best_CI, best_Brier = 0., 1.
    score_CI, score_Brier = [], []
    list_CI, list_Brier = [], []
    patience = 0
    oldloss = 99999

    # Lists for plots
    loss_epoch_train = []
    loss_epoch_C_train = []
    loss_epoch_UC_train = []
    loss_epoch_disc_uc_1_train = []

    loss_epoch_test = []
    loss_epoch_C_test = []
    loss_epoch_UC_test = []
    loss_epoch_disc_uc_1_test = []

    for i in range(num_iter):
        model, loss_epoch, loss_epoch_C, loss_epoch_UC, loss_epoch_disc_uc_1 = train_model_UC_C(
            network_settings, training_settings, model, optimizer, DATA_tr_batch, num_batch_train, var_prev, tr_events, tr_times)

        # train - list loss
        loss_epoch_train.append(loss_epoch)
        loss_epoch_C_train.append(loss_epoch_C)
        loss_epoch_UC_train.append(loss_epoch_UC)
        loss_epoch_disc_uc_1_train.append(loss_epoch_disc_uc_1)
        loss_train.append(loss_epoch.numpy())

        # %%  Test Model UC
        '''Test Model'''
        loss_epoch_te, loss_epoch_te_C, loss_epoch_te_UC, loss_epoch_te_disc_uc_1 = test_model_UC_C(
            network_settings, training_settings, model, DATA_te_batch, num_batch_test, var_prev, tr_events, tr_times)

        # train - list loss
        loss_epoch_test.append(loss_epoch_te)
        loss_epoch_C_test.append(loss_epoch_te_C)
        loss_epoch_UC_test.append(loss_epoch_te_UC)
        loss_epoch_disc_uc_1_test.append(loss_epoch_te_disc_uc_1)
        loss_test.append(loss_epoch_te.numpy())

        # %% assess train-test model_UC
        _, time_test, label_test = DATA_te

        pretrain_state = False
        CI = compute_CI_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state)
        Brier = compute_Brier_scores(model, quantiles, DATA_te, DATA_tr, pretrain_state, risk=0)

        mean_Brier = np.mean(Brier)
        mean_CI = np.mean(CI)


        if mean_CI > best_CI:
            best_CI = mean_CI
            best_CI2 = CI
        if mean_Brier < best_Brier:
            best_Brier = mean_Brier
            best_Brier2 = Brier

        score_CI.append(best_CI)
        list_CI.append(CI)
        list_Brier.append(Brier)
        score_Brier.append(best_Brier)

        # %% print in the console
        if 0 == i:
            print('============================================================================================')
            print('                         Main training %s - optimizing   Loss'%(training_settings['model_state']))
            print('--------------------------------------------------------------------------------------------')
            print('| % completed | current CI | best CI | current Brier | best Brier | loss test | loss train |')
        print_results(training_settings['num_iter_UC'], i, mean_CI, best_CI, mean_Brier, best_Brier, loss_test[-1],
                                       loss_train[-1])


        if (loss_test[-1] - oldloss) > 0:
            patience += 1
            if patience == 30:
                break
        oldloss = loss_test[-1]
        score_list = [best_CI2, best_Brier2, score_CI, score_Brier, list_Brier, list_CI]
        loss_train_list = [loss_train, loss_epoch_train, loss_epoch_C_train, loss_epoch_UC_train, loss_epoch_disc_uc_1_train]
        loss_test_list = [loss_test, loss_epoch_test, loss_epoch_C_test, loss_epoch_UC_test, loss_epoch_disc_uc_1_test]
    return model, score_list, loss_train_list, loss_test_list