import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plt_save_1(model, dir_path, score_list_C, loss_train, loss_test):
    [best_CI2, best_Brier2, score_CI, score_Brier, list_Brier, list_CI] = score_list_C

    fileDir = '/Outputs/MlpType' + str(model.mlp_type) + '_Hidden' + str(model.hidden_size[0]) + '_' + str(
        model.hidden_size[1]) + '_' + model.dist + '_expert' + str(model.num_experts)
    Path(dir_path + fileDir).mkdir(parents=True, exist_ok=True)  # create "Results" folder if not exits

    file1 = open(dir_path + fileDir + '/log.txt', "a")
    file1.write('\n Best CI Score: \t')
    file1.write(str(best_CI2))

    file1.write('\n Best Brier Score: \t')
    file1.write(str(best_Brier2))

    file1.close()

    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(loss_train)
    plt.plot(loss_test)
    plt.legend(['train loss', 'validation loss'])
    plt.xlabel('epoches')
    plt.xlabel('Loss')
    plt.title('Loss function for ' + ' \n Mlp Type=' + str(model.mlp_type) + ' with ' + str(
        model.hidden_size) + ' Hidden (' + str(model.num_experts) + ' ' + model.dist + ') ')
    fig.savefig(dir_path + fileDir + '/Loss.png')
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(score_CI)
    plt.title('average C-Index')
    fig.savefig(dir_path + fileDir + '/AverageCIScore.png')
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(score_Brier)
    plt.title('average Brier Score')
    fig.savefig(dir_path + fileDir + '/AverageBrierScore.png')
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(list_Brier)
    plt.title('Brier Score')
    plt.legend(['quantile1', 'quantile2', 'quantile3', 'quantile4'])
    fig.savefig(dir_path + fileDir + '/BrierScore.png')
    plt.close(fig)

    plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(list_CI)
    plt.title('C-Index')
    plt.legend(['quantile1', 'quantile2', 'quantile3', 'quantile4'])
    fig.savefig(dir_path + fileDir + '/CIScore.png')
    plt.close(fig)


def plt_save_2(training_settings, loss_train_list, loss_test_list, output_path, section):
    fnt_size = 20
    line_width = 5

    data_set = training_settings['dataset']
    loss_method = training_settings['loss_method']

    [loss_train, loss_epoch_train, loss_epoch_C_train, loss_epoch_UC_train, loss_epoch_disc_uc_1_train] = loss_train_list
    [loss_test, loss_epoch_test, loss_epoch_C_test, loss_epoch_UC_test, loss_epoch_disc_uc_1_test] = loss_test_list



    # main loss
    str_plt = '%s_applied_loss_%s_methLossUC%s_methLossC%s' % (section, data_set, loss_method[0], loss_method[1])
    train_vec = np.array(loss_epoch_train)
    test_vec = np.array(loss_epoch_test)
    plt_loss(train_vec, test_vec, output_path, str_plt, fnt_size, line_width)

    # Censored DSM loss
    str_plt = '%s_DSM_CEN_loss_%s_methLossUC%s_methLossC%s' % (section, data_set, loss_method[0], loss_method[1])
    train_vec = np.array(loss_epoch_C_train)
    test_vec = np.array(loss_epoch_C_test)
    plt_loss(train_vec, test_vec, output_path, str_plt, fnt_size, line_width)

    # UnCensored DSM loss
    str_plt = '%s_DSM_UC_loss_%s_methLossUC%s_methLossC%s' % (section, data_set, loss_method[0], loss_method[1])
    train_vec = np.array(loss_epoch_UC_train)
    test_vec = np.array(loss_epoch_UC_test)
    plt_loss(train_vec, test_vec, output_path, str_plt, fnt_size, line_width)

    # Disc loss uc_1
    str_plt = '%s_DISC_UC1_loss_%s_methLossUC%s_methLossC%s' % (section, data_set, loss_method[0], loss_method[1])
    train_vec = np.array(loss_epoch_disc_uc_1_train)
    test_vec = np.array(loss_epoch_disc_uc_1_test)
    plt_loss(train_vec, test_vec, output_path, str_plt, fnt_size, line_width)



def plt_loss(train_loss, test_loss, output_path, str_plt, fnt_size, line_width):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(train_loss, linewidth=line_width)
    plt.plot(test_loss, linewidth=line_width)

    plt.grid()
    plt.legend(['Trian Loss', 'Test Loss'])
    str_tit = 'Loss function --- ' + str_plt
    plt.title(str_tit)
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    # plt.rcParams.update({'font.size': fnt_size})
    # plt.show()
    str_save = output_path + str_plt + '.png'
    fig.savefig(str_save)

def plt_loss_sub(train_loss, valid_loss, fold_num, mlp_type, hidden_size, model_dist, file_name, fnt_size, type):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(train_loss, linewidth=5)
    plt.plot(valid_loss, linewidth=5)

    plt.grid()
    plt.legend(['Trian Loss', 'Validation Loss'])
    plt.title(type + ' Loss function for ' + 'fold =' + str(fold_num) + ' \n Mlp Type=' + str(mlp_type) + ' with ' + str(
        hidden_size) + ' Hidden Node (' + model_dist + ')')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    # plt.rcParams.update({'font.size': fnt_size})
    # plt.show()
    fig.savefig(  file_name +'_'+ type+ '_Loss.png')

def plt_qunatile(CIscores_all_q,  fold_num, mlp_type, hidden_size, model_dist, file_name, fnt_size):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    acc = np.array(CIscores_all_q)
    plt.plot(acc[:, 0], linewidth=5)
    plt.plot(acc[:, 1], linewidth=5)
    plt.plot(acc[:, 2], linewidth=5)
    plt.plot(acc[:, 3], linewidth=5)
    plt.grid()
    plt.legend(['Quantile1', 'Quantile2', 'Quantile3', 'Quantile4'])
    plt.title('Validation Accuracy fold =' + str(fold_num) + ' \n Mlp Type=' + str(mlp_type) + ' with ' + str(
        hidden_size) + ' Hidden Node (' + model_dist + ')')
    plt.xlabel('Epoches')
    plt.ylabel('Acc')
    # plt.rcParams.update({'font.size': fnt_size})
    # plt.show()
    fig.savefig(file_name + '_Acc.png')


def plt_cscore(out, fold_num, mlp_type, hidden_size, model_dist, file_name):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], out)
    plt.xlabel('Quantiles')
    plt.ylabel('C-Score')
    plt.title('Test Acc for Fold=' + str(fold_num) + ' \n Mlp Type=' + str(mlp_type) + ' with ' + str(
        hidden_size) + ' Hidden Node (' + model_dist+ ')')
    fig.savefig(file_name + '_Test.png')

def plt_logit(logits, fold_num, mlp_type, hidden_size, model_dist, file_name):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.bar(np.arange(len(logits)), logits)
    plt.title('Expert Coefs (fold =' + str(fold_num) + ') \nMlp Type=' + str(mlp_type) + ' with ' + str(
        hidden_size) + ' Hidden Node (' + model_dist + ')')
    fig.savefig(file_name + '_Coefs.png')

def plt_brier(times_B, brierScore, fold_num, mlp_type, hidden_size, model_dist, file_name):
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.bar(times_B, brierScore)
    plt.title('Brier Score (fold =' + str(fold_num) + ') \nMlp Type=' + str(mlp_type) + ' with ' + str(
        hidden_size) + ' Hidden Node (' + model_dist + ')')
    fig.savefig(file_name + '_Brier.png')



def plt_final_res(finalRes, mlp_type, hidden_size, model_dist, file_name):
    fig = plt.figure()
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], finalRes)
    plt.xlabel('Quantiles')
    plt.ylabel('C-Score')
    plt.title('Test Acc for All Folds Mlp Type=' + str(mlp_type) + ' with ' + str(hidden_size) + ' Hidden Node (' + model_dist + ')')
    fig.savefig(file_name + '_AllFoldsTest.png')
    
def Visualize_Dataset(times,events,numberofSamples,path):
    idx_uncensored = np.where(np.array(events[0:numberofSamples])==1)
    idx_censored = np.where(np.array(events[0:numberofSamples])==0)
    
    x_arr = np.array(times[0:numberofSamples])
    y_arr = np.arange(0,len(x_arr))
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    plt.hlines(y_arr, 0, x_arr, color='black')  # Stems
    
    plt.plot(x_arr[idx_censored], y_arr[idx_censored], 'D', color='red')  # Stem ends
    plt.plot(x_arr[idx_uncensored], y_arr[idx_uncensored], 'D', color='green')  # Stem ends
    plt.legend(['censored','uncensored'])
    
    
    plt.plot([0, 0], [y_arr.min(), y_arr.max()], '--'); # Middle bar
    
    idx_uncensored = np.where(np.array(events)==1)
    
    sp = 100*(1-len(idx_uncensored[0])/len(times))
    
    plt.title('Data from Model 1 with %.2f%% Censored 1'%sp)
    
    fig.savefig(path + 'dataset.png')