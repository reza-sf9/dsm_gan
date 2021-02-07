import numpy as np
import tensorflow as tf
from tensorflow import keras


_EPSILON = 1e-08


class deep_survival_machines_pretrain(keras.Model):
    '''this class used for pretarining the model'''
    def __init__(self, network_settings):
        super(deep_survival_machines_pretrain, self).__init__()

        # self.adj= adj

        self.num_experts = network_settings['num_experts']
        self.mlp_type = network_settings['mlp_type']
        self.dist = network_settings['distribution']
        self.hidden_size = network_settings['hidden_size']
        self.init = network_settings['init']
        self.input_dim = network_settings['dim_input']

        # RNN Network parameterts
        self.kernel_initializer = network_settings['kernel_initializer']
        self.kernel_regularizer = network_settings['kernel_regularizer']
        self.activity_regularizer = network_settings['activity_regularizer']

        # Layers

        if self.dist == 'Weibull':
            self.act = tf.keras.activations.selu
            if self.init is not False:
                self.shape = tf.Variable(tf.cast(self.init[0] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='scaleVar')
                self.scale = tf.Variable(tf.cast(self.init[1] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='shapeVar')
            else:
                self.scale = tf.Variable(-tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='scaleVar')
                self.shape = tf.Variable(-tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='shapeVar')
        elif self.dist == 'LogNormal':
            self.act = tf.keras.activations.tanh
            if self.init is not False:
                self.shape = tf.Variable(tf.cast(self.init[0] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='scaleVar')
                self.scale = tf.Variable(tf.cast(self.init[1] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='shapeVar')
            else:
                self.scale = tf.Variable(tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='scaleVar')
                self.shape = tf.Variable(tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='shapeVar')

        if self.mlp_type == 1:
            self.gate = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=False, dtype='float64'))
            self.scaleg = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=True, dtype='float64'))
            self.shapeg = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=True, dtype='float64'))

        if self.mlp_type == 2:
            # self.gate = keras.Sequential([keras.layers.Dense(num_experts , input_shape=(hidden_size[0],) , use_bias=False, dtype='float64',name='mlp2_gate_Dense'),
            #                              keras.layers.Softmax()],name='mlp2_gateNet')
            self.gate = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=False,
                                                             dtype='float64', name='mlp2_gate_Dense')], name='mlp2_gateNet')
            self.scaleg = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=True,
                                                               dtype='float64', name='mlp2_scale_Dense')], name='mlp2_scaleNet')
            self.shapeg = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=True,
                                                               dtype='float64', name='mlp2_shape_Dense')], name='mlp2_shapeNet')
            self.embedding = keras.Sequential([
                keras.layers.Dense(self.hidden_size[0], use_bias=False, dtype='float64', name='mlp2_embed_Dense1'),
                keras.layers.Activation(tf.nn.relu6, name='mlp2_Embed_Relu1')], name='mlp2_embedding_Net')

    def call(self, x, pre_training=False):

        ''' Traditional DSM'''

        if self.mlp_type == 1:
            embed = x
        else:
            embed = self.embedding(x)

        if (pre_training):
            shape = self.shape
            scale = self.scale
            logits = self.gate(embed) / 1000
        else:
            shape = self.act(self.shapeg(embed)) + tf.repeat(tf.expand_dims(tf.cast(self.shape, 'float64'), 0), x.shape[0], 0)
            scale = self.act(self.scaleg(embed)) + tf.repeat(tf.expand_dims(tf.cast(self.scale, 'float64'), 0), x.shape[0], 0)
            logits = self.gate(embed) / 1000

        return [shape, scale, logits]


class deep_survival_machines(keras.Model):
    '''this class used for training UC and Cen model'''
    def __init__(self, network_settings):
        super(deep_survival_machines, self).__init__()

        # self.adj= adj

        self.num_experts = network_settings['num_experts']
        self.mlp_type = network_settings['mlp_type']
        self.dist = network_settings['distribution']
        self.hidden_size = network_settings['hidden_size']
        self.init = network_settings['init']
        self.input_dim = network_settings['dim_input']

        self.hidden_size_dsm = network_settings['hidden_size_dsm']
        self.hidden_size_gan_x = network_settings['hidden_size_gan_x']
        self.hidden_size_gan_t = network_settings['hidden_size_gan_t']
        self.use_bias_status = network_settings['use_bias']

        # RNN Network parameterts
        self.kernel_initializer = network_settings['kernel_initializer']
        self.kernel_regularizer = network_settings['kernel_regularizer']
        self.activity_regularizer = network_settings['activity_regularizer']

        '''DSM LAYERS'''
        # Layers
        if self.dist == 'Weibull':
            self.act = tf.keras.activations.selu
            if self.init is not False:
                self.shape = tf.Variable(tf.cast(self.init[0] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='scaleVar')
                self.scale = tf.Variable(tf.cast(self.init[1] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='shapeVar')
            else:
                self.scale = tf.Variable(-tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='scaleVar')
                self.shape = tf.Variable(-tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='shapeVar')
        elif self.dist == 'LogNormal':
            self.act = tf.keras.activations.tanh
            if self.init is not False:
                self.shape = tf.Variable(tf.cast(self.init[0] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='scaleVar')
                self.scale = tf.Variable(tf.cast(self.init[1] * tf.ones(self.num_experts, 'float64'), 'float64'), trainable=True,
                                         dtype='float64', name='shapeVar')
            else:
                self.scale = tf.Variable(tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='scaleVar')
                self.shape = tf.Variable(tf.ones(self.num_experts, dtype='float64'), trainable=True, dtype='float64', name='shapeVar')

        if self.mlp_type == 1:
            self.gate = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=False, dtype='float64'))
            self.scaleg = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=True, dtype='float64'))
            self.shapeg = keras.Sequential(keras.layers.Dense(self.num_experts, input_shape=(self.input_dim,), use_bias=True, dtype='float64'))

        if self.mlp_type == 2:
            # self.gate = keras.Sequential([keras.layers.Dense(num_experts , input_shape=(hidden_size[0],) , use_bias=False, dtype='float64',name='mlp2_gate_Dense'),
            #                              keras.layers.Softmax()],name='mlp2_gateNet')
            self.gate = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=False,
                                                             dtype='float64', name='mlp2_gate_Dense')], name='mlp2_gateNet')
            self.scaleg = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=True,
                                                               dtype='float64', name='mlp2_scale_Dense')], name='mlp2_scaleNet')
            self.shapeg = keras.Sequential([keras.layers.Dense(self.num_experts, input_shape=(self.hidden_size[0],), use_bias=True,
                                                               dtype='float64', name='mlp2_shape_Dense')], name='mlp2_shapeNet')
            self.embedding = keras.Sequential([
                keras.layers.Dense(self.hidden_size[0], use_bias=False, dtype='float64', name='mlp2_embed_Dense1'),
                keras.layers.Activation(tf.nn.relu6, name='mlp2_Embed_Relu1')], name='mlp2_embedding_Net')

        '''DEFINE THE MODEL OF DISCRIMINATOR'''
        # the NN of x
        l_x_disc = len(self.hidden_size_gan_x)
        if l_x_disc == 2:
            self.disc_x = keras.Sequential([tf.keras.layers.Dense(self.hidden_size_gan_x[0], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status),
                                            tf.keras.layers.Dense(self.hidden_size_gan_x[1], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status)])
        elif l_x_disc == 1:
            self.disc_x = keras.Sequential([tf.keras.layers.Dense(self.hidden_size_gan_x[0], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status)])

        # the NN of t
        l_t_disc = len(self.hidden_size_gan_t)
        if l_t_disc == 2:
            self.disc_t = keras.Sequential([tf.keras.layers.Dense(self.hidden_size_gan_t[0], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status),
                                            tf.keras.layers.Dense(self.hidden_size_gan_t[1], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status)])

        elif l_t_disc == 1:
            self.disc_t = keras.Sequential([tf.keras.layers.Dense(self.hidden_size_gan_t[0], activation=tf.nn.relu6,
                                                                  use_bias=self.use_bias_status)])

        # NN of concatenate t and x
        self.disc_x_t = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=self.use_bias_status)])

        if self.init is not False:
            self.shape = tf.cast(np.random.normal(size=self.num_experts), 'double')
            self.scale = tf.cast(np.random.normal(size=self.num_experts), 'double')

    def call(self, x, t, pre_training=False):
        t = tf.convert_to_tensor(t)
        t = tf.expand_dims(t, 1)

        ''' Traditional DSM'''

        if self.mlp_type == 1:
            embed = x
        else:
            embed = self.embedding(x)

        out_shape = self.act(self.shapeg(embed)) + tf.repeat(tf.expand_dims(tf.cast(self.shape, 'float64'), 0), x.shape[0], 0)
        out_scale = self.act(self.scaleg(embed)) + tf.repeat(tf.expand_dims(tf.cast(self.scale, 'float64'), 0), x.shape[0], 0)
        out_gate = self.gate(embed) / 1000

        # out_1_1 = self.act(self.shapeg(embed))  # batch* 8
        # out_1_2 = tf.repeat(tf.expand_dims(tf.cast(self.shape, 'double'), 0), x.shape[0], 0)  # batch* 8
        # out_shape = out_1_1 + out_1_2  # batch* 8
        #
        # out_2_1 = self.act(self.scaleg(embed))
        # out_2_2 = tf.repeat(tf.expand_dims(tf.cast(self.scale, 'double'), 0), x.shape[0], 0)
        # out_scale = out_2_1 + out_2_2
        #
        # out_gate = self.gate(embed) / 1000

        '''Discriminator'''
        mean_dist_log_n = tf.math.exp(out_scale + tf.math.divide(tf.math.pow(out_shape, 2), 2))
        alpha = tf.nn.softmax(out_gate)
        t_fake = tf.reduce_mean(tf.math.multiply(mean_dist_log_n, alpha), 1)
        t_fake = tf.expand_dims(t_fake, 1)

        out_disc_x = self.disc_x(x)

        out_disc_t_real = self.disc_t(t)  # t should be passed as a tensorflow
        out_disc_t_fake = self.disc_t(t_fake)

        conc_dsic_x_t_real = tf.concat([out_disc_x, out_disc_t_real], axis=1)
        conc_disc_x_t_fake = tf.concat([out_disc_x, out_disc_t_fake], axis=1)

        out_disc_x_t_real = self.disc_x_t(conc_dsic_x_t_real)
        out_disc_x_t_fake = self.disc_x_t(conc_disc_x_t_fake)

        out_disc_x_t_real_1 = tf.keras.activations.sigmoid(out_disc_x_t_real)
        out_disc_x_t_fake_1 = tf.keras.activations.sigmoid(out_disc_x_t_fake)

        self.outputs = (out_shape, out_scale, out_gate, out_disc_x_t_real_1, out_disc_x_t_fake_1)
        shape, scale, logits, disc_x_t_real, disc_x_t_fake = self.outputs

        return self.outputs



def dsm_weibul_loss(shape, scale, logits, t, e, ELBO, lambda_, alpha):
    ''' LOSS-FUNCTION 1: Weibul loss '''
    # shape = shape.detach().cpu().numpy()
    # scale = scale.detach().cpu().numpy()
    # logits = logits.detach().cpu().numpy()

    G = scale.shape[1]

    # t = t_test.numpy()
    # e = e_test.numpy()

    t = tf.cast(t, 'float64')
    logits = tf.cast(logits, 'float64')
    # print shape, scale, logits
    lossf = []  # pdf
    losss = []  # survival

    k_ = tf.cast(shape, 'float64')
    b_ = tf.cast(scale, 'float64')

    for g in range(G):
        k = k_[:, g]
        b = b_[:, g]

        f = k + b + ((tf.math.exp(k) - 1) * (b + tf.math.log(t + 1e-14))) - (tf.math.pow(tf.math.exp(b) * t, tf.math.exp(k)))

        s = - (tf.math.pow(tf.math.exp(b) * t, tf.math.exp(k)))

        b_exp = tf.math.exp(-b)  # b_exp = scale
        k_exp = tf.math.exp(-k)  # k_exp = 1/shape

        lossf.append(f)
        losss.append(s)

    losss = tf.stack(losss, axis=1)
    lossf = tf.stack(lossf, axis=1)

    if ELBO:

        lossg = tf.nn.softmax(logits, axis=1)
        losss = lossg * losss
        lossf = lossg * lossf

        losss = tf.math.reduce_sum(losss, axis=1)
        lossf = tf.math.reduce_sum(lossf, axis=1)

    else:
        lossg = tf.nn.log_softmax(logits, axis=1)
        losss = lossg + losss
        lossf = lossg + lossf

        losss = tf.math.reduce_logsumexp(losss + 1e-14, axis=1)
        lossf = tf.math.reduce_logsumexp(lossf + 1e-14, axis=1)

    lossg = tf.nn.softmax(logits, axis=1)

    uncensored = tf.squeeze(tf.where(e == 1))
    censored = tf.squeeze(tf.where(e == 0))

    loss_Uncensored = tf.math.reduce_sum(tf.gather(lossf, uncensored))
    loss_censored = alpha * tf.math.reduce_sum(tf.gather(losss, censored))

    loss = loss_Uncensored + loss_censored

    if (len(uncensored.shape) > 0):
        a = uncensored.shape
    else:
        a = 1
    a = tf.cast(a, 'float64')

    return -loss / t.shape[0], loss_Uncensored / a, loss_censored / censored.shape[0]


''' LOSS-FUNCTION 2: lognormal loss '''


def __dsm_loss_log_normal_artcl(t, uncensored, censored, scale, shape, logits,  ELBO, alpha):
    '''dsm loss function - based on Likelihood'''
    if len(shape.shape) == 1:
        shape = tf.expand_dims(shape, axis=0)
        scale = tf.expand_dims(scale, axis=0)

    G = scale.shape[1]
    lossf = []  # pdf
    losss = []  # survival
    lossm = []  # sum of squared error of mean / median survival time

    k_ = tf.cast(shape, 'float64')
    b_ = tf.cast(scale, 'float64')

    t = tf.cast(tf.squeeze(t), 'float64')

    for g in range(G):
        mu = k_[:, g]
        sigma = b_[:, g]

        # article loss
        f = - sigma - 0.5 * tf.cast(np.log(2 * np.pi), 'float64') - tf.math.divide((tf.math.log(t) - mu) ** 2, 2. * tf.math.exp(2 * sigma))

        s = tf.math.divide(tf.math.log(t) - mu, tf.math.exp(sigma) * tf.cast(np.sqrt(2), 'float64'))
        s = 0.5 - 0.5 * tf.math.erf(s)
        s = tf.math.log(s + 1e-7)

        lossf.append(f)
        losss.append(s)

    losss = tf.stack(losss, axis=1)
    lossf = tf.stack(lossf, axis=1)

    if ELBO:
        lossg = tf.nn.softmax(logits, axis=1)
        # lossg = logits
        losss = lossg * losss
        lossf = lossg * lossf
        losss = tf.math.reduce_sum(losss, axis=1)
        lossf = tf.math.reduce_sum(lossf, axis=1)

    else:
        lossg = tf.nn.log_softmax(logits, axis=1)
        # lossg = logits
        losss = lossg + losss
        lossf = lossg + lossf
        losss = tf.math.reduce_logsumexp(losss, axis=1)
        lossf = tf.math.reduce_logsumexp(lossf, axis=1)


    dsm_loss_uc = tf.math.reduce_sum(tf.gather(lossf, uncensored))
    dsm_loss_cen = alpha * tf.math.reduce_sum(tf.gather(losss, censored))

    return dsm_loss_uc, dsm_loss_cen

def __dsm_mse_loss_uc(shape, scale, logits, t, uncensored, censored):
    ''' dsm loss based on mse'''
    t_uc = tf.gather(t, uncensored)
    shape_uc = tf.gather(shape, uncensored)
    scale_uc = tf.gather(scale, uncensored)
    logits_uc = tf.gather(logits, uncensored)

    mu_uc = shape_uc
    sigma = scale_uc
    sigma_2_uc = tf.math.exp(sigma*2)
    alpha_uc = tf.nn.softmax(logits_uc, axis=1)

    # finding mean and its loss
    mean_uc = tf.math.exp(mu_uc + tf.math.divide(sigma_2_uc, 2)) * alpha_uc
    mean_uc = tf.reduce_mean(mean_uc, axis=1)
    l_mese_uc = tf.pow((mean_uc - t_uc), 2)

    # finding var
    var_uc = (tf.math.exp(sigma_2_uc) - 1) * np.exp(2*mu_uc + sigma_2_uc)
    var_uc = var_uc * (alpha_uc ** 2)
    var_uc = tf.reduce_mean(var_uc, axis=1)


    l_mean_uc = tf.reduce_mean(l_mese_uc) / uncensored.shape[0]
    l_var_uc = tf.math.reduce_sum(var_uc) / uncensored.shape[0]

    loss = l_mean_uc + l_var_uc
    return loss

def __dsm_mse_loss_cen_1(shape, scale, logits, t, uncensored, censored,  var_, e_tot, t_train):
    '''mse loss for Censored data of dsm NN,
    for variance, I use the half of distance between censored time and time of last measurement
    '''

    mu = shape
    sigma = scale
    sigma_2 = tf.math.exp(sigma*2)
    alpha = tf.nn.softmax(logits, axis=1)

    l2 = tf.math.exp(mu + tf.math.divide(sigma_2, 2)) * alpha
    l2 = tf.reduce_sum(l2, axis=1)
    l2 = tf.pow((l2 - t), 2)


    l_var = (tf.math.exp(sigma_2) - 1) * np.exp(2*mu + sigma_2)
    l_var = l_var * (alpha ** 2)
    l_var = tf.reduce_sum(l_var, axis=1)

    loss_UC = tf.math.reduce_sum(tf.gather(l2, uncensored)) / uncensored.shape[0]
    loss_UC_var = tf.math.reduce_sum(tf.gather(l_var, uncensored)) / uncensored.shape[0]

    loss_C = tf.math.reduce_sum(tf.gather(l2, uncensored)) / uncensored.shape[0]
    loss_C_var = tf.math.reduce_sum(tf.gather(l_var, uncensored)) / uncensored.shape[0]


    loss = loss_UC + loss_UC_var
    return loss

def __dsm_mse_loss_cen_2(shape, scale, logits, t, uncensored, censored,  var_tot, e_tot, t_train):
    '''mse less for cnesored based on variance of uncensored data'''

    t_cen = tf.gather(t, censored)
    shape_cen = tf.gather(shape, censored)
    scale_cen = tf.gather(scale, censored)
    logits_cen = tf.gather(logits, censored)

    # finding mean
    mu_c = shape_cen
    sigma_c = scale_cen
    sigma_2_c = tf.math.exp(sigma_c*2)
    alpha_c = tf.nn.softmax(logits_cen, axis=1)

    mean_c = tf.math.exp(mu_c + tf.math.divide(sigma_2_c, 2)) * alpha_c
    mean_c = tf.reduce_mean(mean_c, axis=1)
    l_mean_c = tf.keras.losses.mean_squared_error(mean_c, t_cen)
    l_abs_c = tf.keras.losses.mean_absolute_error(mean_c, t_cen)

    # finding var
    m = scale.shape
    var_selected = np.zeros((len(censored), m[1]))

    for i in range(len(t_cen)):
        ind_up = tf.squeeze(tf.where(t_cen[i] < t_train))
        temp_var = tf.gather(var_tot, ind_up)
        var_selected[i, :] = tf.reduce_mean(temp_var, 0)

    var_c = var_selected
    var_c = var_c * (alpha_c ** 2)
    l_var_c = tf.reduce_mean(var_c, axis=1)

    loss_mean_C = l_mean_c / censored.shape[0]
    loss_var_C = tf.reduce_mean(l_var_c) / censored.shape[0]
    l_abs_c = l_abs_c / censored.shape[0]

    # loss = loss_mean_C + loss_var_C
    loss = l_abs_c
    return loss

'''discriminator'''
def __disc_loss_log_normal(disc_x_t_real, disc_x_t_gen, shape, scale, logits, t, censored, uncensored):
    # calculate t_gen from mean of observations

    t_real = t

    #NEW
    mu = shape
    sigma = scale
    sigma_2 = tf.math.exp(sigma * 2)
    alpha = tf.nn.softmax(logits, axis=1)

    mean_dsit = tf.math.exp(mu + tf.math.divide(sigma_2, 2)) * alpha

    t_gen = tf.reduce_mean(mean_dsit, 1)
    t_gen = tf.expand_dims(t_gen, 1)

    # seperate censored & uncensored
    t_real_cen = tf.gather(t_real, censored)
    t_real_uc = tf.gather(t_real, uncensored)

    t_gen_uc = tf.gather(t_gen, uncensored)
    t_gen_cen = tf.gather(t_gen, censored)

    disc_x_t_real_uc = tf.gather(disc_x_t_real, uncensored)
    disc_x_t_real_cen = tf.gather(disc_x_t_real, censored)

    disc_x_t_gen_uc = tf.gather(disc_x_t_gen, uncensored)
    disc_x_t_gen_cen = tf.gather(disc_x_t_gen, censored)

    # main loss
    a = tf.math.log(disc_x_t_real_uc)
    aa = -tf.reduce_mean(a)

    b = tf.math.log(1 - disc_x_t_gen_uc + 1e-7)
    bb = -tf.reduce_mean(b)
    disc_loss_uc_1 = aa + bb

    # get error here
    mse_loss = tf.keras.losses.MSE(t_real_uc, t_gen_uc)
    disc_loss_uc_2 = tf.reduce_mean(mse_loss)

    # disc_loss_cen_1 = -tf.reduce_mean()
    c = tf.math.maximum(1e-7, t_gen_cen - t_real_cen)
    disc_loss_cen_1 = tf.reduce_mean(c)

    return disc_loss_uc_1, disc_loss_uc_2, disc_loss_cen_1

def dsm_log_normal_loss(training_settings, shape, scale, logits, disc_x_t_real, disc_x_t_gen, t, e, var_, e_tot, t_train):
    ELBO = training_settings['ELBO']
    lambda_ = training_settings['lambda']
    alpha = training_settings['alpha']
    loss_method = training_settings['loss_method']
    model_state = training_settings['model_state']

    e = tf.squeeze(e)
    uncensored = tf.squeeze(tf.where(e == 1))
    censored = tf.squeeze(tf.where(e == 0))

    # call dsm loss - article
    dsm_loss_uc_artcl, dsm_loss_cen_artcl = __dsm_loss_log_normal_artcl(t, uncensored, censored, scale, shape, logits,  ELBO, alpha)
    dsm_loss_uc_artcl = -dsm_loss_uc_artcl / uncensored.shape[0]
    dsm_loss_cen_artcl = -dsm_loss_cen_artcl / censored.shape[0]

    # call dsm loss - mse
    dsm_loss_uc_mse = __dsm_mse_loss_uc(shape, scale, logits, t, uncensored, censored)
    dsm_loss_uc_mse = dsm_loss_uc_mse / uncensored.shape[0]

    # call discriminator loss
    disc_loss_uc_1, disc_loss_uc_2, disc_loss_cen_1 = __disc_loss_log_normal(disc_x_t_real, disc_x_t_gen,
                                                                           shape, scale, logits, t,
                                                                           censored, uncensored)
    disc_loss_uc_1 = disc_loss_uc_1/uncensored.shape[0]
    disc_loss_uc_2 = disc_loss_uc_2/uncensored.shape[0]
    disc_loss_cen_1 = disc_loss_cen_1/censored.shape[0]


    if (len(uncensored.shape) > 0):
        a = uncensored.shape
    else:
        a = 1
    a = tf.cast(a, 'float64')

    if model_state == 'UC':
        loss_method = loss_method[0]
        if loss_method == 1:
            loss = dsm_loss_uc_artcl

        elif loss_method == 2:
            loss = dsm_loss_uc_artcl + dsm_loss_cen_artcl

        elif loss_method == 3:
            loss = dsm_loss_cen_artcl

        elif loss_method == 4:
            loss = dsm_loss_uc_mse

        elif loss_method == 5:
            dsm_loss = dsm_loss_uc_artcl
            disc_loss = disc_loss_uc_1
            loss = disc_loss

        elif loss_method == 6:
            dsm_loss = dsm_loss_uc_artcl + dsm_loss_cen_artcl
            disc_loss = disc_loss_uc_1 + disc_loss_uc_2
            loss = dsm_loss + disc_loss

    elif model_state == 'Cen':
        loss_method = loss_method[1]

        if loss_method==1: # Navid's method
            loss_cen_1 = __dsm_mse_loss_cen_1(shape, scale, logits, t, uncensored, censored,  var_, e_tot, t_train)
            loss = loss_cen_1

        elif loss_method==2: # Nick's method
            loss_cen_2 = __dsm_mse_loss_cen_2(shape, scale, logits, t, uncensored, censored,  var_, e_tot, t_train)
            loss = loss_cen_2

    return loss, dsm_loss_uc_artcl, dsm_loss_cen_artcl, disc_loss_uc_1


'''
This loss function is used for pre training model. 
in pretraining we don't use weigted usm of experts
'''
def _logNormalLoss_pretrain(network_settings_pretrain, shape, scale, logits, t, e, method_loss):
    uncensored = tf.squeeze(tf.where(e == 1))
    censored = tf.squeeze(tf.where(e == 0))
    ELBO = network_settings_pretrain['ELBO']
    alpha = network_settings_pretrain['alpha']

    # article method
    if method_loss == 1:
        dsm_loss_uc, dsm_loss_cen = __dsm_loss_log_normal_artcl(t, uncensored, censored, scale, shape, logits, ELBO, alpha)
        loss_tot = dsm_loss_uc + dsm_loss_cen

    # mse method
    elif method_loss == 2:
        mse_loss_uc = __dsm_mse_loss_uc(shape, scale, logits, t, uncensored, censored)
        loss_tot = mse_loss_uc


    return -tf.reduce_mean(loss_tot)


# fitting weibull using only t and e
def _weibullLoss(shape, scale, logits, t, e):
    # k_ = shape # mu
    # b_ = scale # logVar (scale = log(s^2)  => s = exp(0.5*scale))

    # shape = shape.detach().cpu().numpy()
    # scale = scale.detach().cpu().numpy()
    if len(shape.shape) == 1:
        shape = tf.expand_dims(shape, axis=0)
        scale = tf.expand_dims(scale, axis=0)
    # mu
    k_ = tf.cast(tf.tile(shape, (t.shape[0], 1)), 'float64')
    # logVar (scale = log(s^2)  => s = exp(0.5*scale))
    b_ = tf.cast(tf.tile(scale, (t.shape[0], 1)), 'float64')

    ll = 0.

    G = k_.shape[1]
    ll = 0

    for g in range(G):
        k = k_[:, g]
        b = b_[:, g]

        f = k + b + ((tf.math.exp(k) - 1) * (b + tf.math.log(t))) - (tf.math.pow(tf.math.exp(b) * t, tf.math.exp(k)))
        s = - (tf.math.pow(tf.math.exp(b) * t, tf.math.exp(k)))

        uncens = np.where(e == 1)[0]
        cens = np.where(e == 0)[0]

        uncensored = tf.squeeze(tf.where(e == 1))
        censored = tf.squeeze(tf.where(e == 0))

        loss_Uncensored = tf.math.reduce_sum(tf.gather(f, uncensored))
        loss_censored = tf.math.reduce_sum(tf.gather(s, censored))

        loss = loss_Uncensored + loss_censored
        # loss = loss_Uncensored

        ll += loss

    return -tf.reduce_mean(ll)


def dsm_mse_loss(shape, scale, logits, t, e, ELBO, lambda_, alpha):
    G = scale.shape[1]

    k_ = tf.cast(shape, 'float64')
    b_ = tf.cast(scale, 'float64')

    gate = tf.nn.softmax(logits, axis=1)

    l2 = tf.math.exp(shape + tf.math.exp(scale) / 2) * gate
    l2 = tf.reduce_sum(l2, axis=1)
    l2 = tf.pow((l2 - t), 2)

    sigma = tf.math.exp(0.5 * scale)

    l_var = (tf.math.exp(sigma ** 2) - 1) * np.exp(2 * shape + sigma ** 2)
    l_var = l_var * (gate ** 2)
    l_var = tf.reduce_sum(l_var, axis=1)

    uncensored = tf.squeeze(tf.where(e == 1))
    censored = tf.squeeze(tf.where(e == 0))

    loss_Uncensored = tf.math.reduce_sum(tf.gather(l2, uncensored)) / uncensored.shape[0]
    loss_Uncensored_var = tf.math.reduce_sum(tf.gather(l_var, uncensored)) / uncensored.shape[0]
    loss_Uncensored_var = loss_Uncensored_var / 100000
    
    loss = loss_Uncensored + loss_Uncensored_var
    return loss


def loss_total(network_settings, training_settings, shape, scale, logits, disc_x_t_real, disc_x_t_fake, t, e,
               var_, e_tot, t_train):
    if network_settings['distribution'] == 'Weibull':
        loss, loss_Uncensored, loss_censored = dsm_weibul_loss(training_settings, shape, scale, logits, t, e)
    elif network_settings['distribution'] == 'LogNormal':
        loss, loss_Uncensored, loss_censored, disc_loss_uc_1 = dsm_log_normal_loss(training_settings, shape, scale, logits, disc_x_t_real,
                                                                                   disc_x_t_fake, t, e,
                                                                                    var_, e_tot, t_train)

    # lossmse = dsm_mse_loss(shape, scale, logits , t, e, ELBO , lambda_, alpha )

    # loss = loss_censored + lossmse
    return loss, loss_Uncensored, loss_censored, disc_loss_uc_1


def pretrain_loss(network_settings_pretrain, shape, scale, logits, mb_times, mb_events):
    if network_settings_pretrain['distribution'] == 'Weibull':
        loss = _weibullLoss(shape, scale, logits, mb_times, mb_events)
    elif network_settings_pretrain['distribution'] == 'LogNormal':
        method_loss = network_settings_pretrain['method_loss']
        loss = _logNormalLoss_pretrain(network_settings_pretrain, shape, scale, logits, mb_times, mb_events, method_loss)

    return loss
