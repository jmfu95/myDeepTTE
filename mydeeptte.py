import os
import tensorflow as tf
import numpy as np

EPS = 10


class EntireEstimator(object):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(EntireEstimator, self).__init__()

        self.input2hid = tf.layers.dense(input_size, hidden_size)

        self.residuals = list()
        for i in range(num_final_fcs):
            self.residuals.append(tf.layers.dense(hidden_size, hidden_size))

        self.hid2out = tf.layers.dense(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = concat((attr_t, sptm_t), dim=1)

        hidden = leaky_relu(self.input2hid(inputs))

        for i in range(len(self.residuals)):
            residual = leaky_relu(self.residuals[i](hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, label, mean, std):
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = tf.abs(pred - label) / label

        return {'label': label, 'pred': pred}, loss.mean()


class LocalEstimator(object):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = tf.layers.dense(input_size, 64)
        self.hid2hid = tf.layers.dense(64, 32)
        self.hid2out = tf.layers.dense(32, 1)

    def forward(self, sptm_s):
        hidden = leaky_relu(self.input2hid(sptm_s))

        hidden = leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first=True)[0]
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = tf.abs(pred - label) / (label + EPS)

        return loss.mean()


class DeepTTeModel(object):

    def __init__(self):
        self.num_classes = num_classes

    def model(self, traj, driverid, weatherid, timeid):
        with tf.variable_scope('scale1'):
            attr_t = attribute(driverid, weatherid, timeid)
            sptm_s, sptm_t = spatio_temporal(self, traj, attr_t)
            entire_estimate = EntireEstimator(input_size=self.spatio_temporal.out_size() + self.attr_net.out_size(),
                                              num_final_fcs=self.num_final_fcs, hidden_size=self.final_fc_size)
            entire_out = entire_estimate.forward(attr_t, sptm_t)
            local_estimate = LocalEstimator(input_size=self.spatio_temporal.out_size())
            for i in range(len(sptm_s)):
                local_out = local_estimate(sptm_s[i])

        return local_out, entire_out


    def attribute(self, driverID, weatherID, TimeID):
        driverid = embedding(driverID)
        weatherid = embedding(weatherID)
        timeid = embedding(TimeID)
        embed_dims = [driverid, weatherid, timeid]
        embed = getattr(self, embed_dims)
        attr_t = attr[name].view(-1, 1)

        em_list.append(attr_t)

        dist = utils.normalize(attr['dist'], 'dist')
        attr_t = em_list.append(dist.view(-1, 1))

        return attr_t

    def spatio_temporal(self, traj, attr_out):
        conv_locs = geoconv(traj)
        expand = expand(conv_locs)
        conv_locs = concate(conv_locs, expand)
        lens = map(lambda x: x - self.kernel_size + 1, traj['lens'])

        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first=True)

        sptm_s, (h_n, c_n) = LSTM(packed_inputs)
        
        '''LSTM(input_size = num_filter + 1 + attr_size, \
                                      hidden_size = 128, \
                                      num_layers = 2, \
                                      batch_first = True
            )
       '''
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(sptm_s, batch_first=True)
        sptm_t = attent_pooling(hiddens, lens, attr_t)

        return sptm_s, sptm_t

#
    def attent_pooling(self, hiddens, lens, attr_t):
        attent = tf.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)
        alpha = tf.matmul(hiddens, attent)
        alpha = tf.exp(-alpha)


        alpha = alpha / tf.reduce_sum(alpha, dim=1, keepdim=True)

        hiddens = hiddens.permute(0, 2, 1)
        hiddens = tf.matmul(hiddens, alpha)
        hiddens = tf.squeeze(hiddens)

        return hiddens





    def loss(self, attr, traj, config):
        pred_dict, entire_loss = self.entire_estimate.eval_on_batch(entire_out, attr['time'], config['time_mean'], config['time_std'])
        local_label = utils.get_local_seq(traj['time_gap'], self.kernel_size, mean, std)
        local_loss = self.local_estimate.eval_on_batch(local_out, local_length, local_label, mean, std)
        self.loss = (1 - self.alpha) * entire_loss + self.alpha * local_loss
        return self.loss

    def optimize(self, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        return train_op







