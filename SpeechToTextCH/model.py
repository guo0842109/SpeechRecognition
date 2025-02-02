#-*- coding:utf-8 -*-
__author__ = 'Deeper'
import tensorflow as tf  # 1.0.0
import numpy as np

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class Model():
    def __init__(self, n_out, batch_size=1, n_mfcc=20, is_training=True):
        n_dim = 128
        self.is_training = is_training

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, n_mfcc])
        self.seq_len = tf.reduce_sum(tf.cast(tf.not_equal(tf.reduce_sum(self.input_data, reduction_indices=2), 0.), tf.int32), reduction_indices=1)
        self.targets = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

        # 1D convolution
        self.conv1d_index = 0
        with tf.variable_scope('conv_layer1'):
            out = self.conv1d_layer(self.input_data, dim=n_dim)
        
        # stack hole CNN
        n_blocks = 3
        skip = 0
        self.aconv1d_index = 0
        with tf.variable_scope('ResNet'):
            for itr_nb in range(n_blocks):
                with tf.variable_scope('ResNet%d'%itr_nb):
                    for r in [1, 2, 4, 8, 16]:
                        out, s = self.residual_block(out, size=7, rate=r, dim=n_dim)
                        skip += s
        with tf.variable_scope('logit_out1'):
            logit = self.conv1d_layer(skip, dim=skip.get_shape().as_list()[-1])
        with tf.variable_scope('logit_out2'):
            self.logit = self.conv1d_layer(logit, dim=n_out, bias=True, activation=None)
        #batch_norm(name='d_bn2')
        # CTC loss
        indices = tf.where(tf.not_equal(tf.cast(self.targets, tf.float32), 0.))
        target = tf.SparseTensor(indices=indices, values=tf.gather_nd(self.targets, indices)-1, dense_shape=tf.cast(tf.shape(self.targets), tf.int64))
        print(tf.shape(self.logit))
        loss = tf.nn.ctc_loss(target, self.logit, self.seq_len, time_major=False, ignore_longer_outputs_than_inputs=True)
        self.cost = tf.reduce_mean(loss)

        # optimizer
        optimizer = tf.train.AdamOptimizer()
        #var_list = [var for var in tf.trainable_variables()]
        #gradient = optimizer.compute_gradients(self.cost, var_list=var_list)
        #self.optimizer_op = optimizer.apply_gradients(gradient)
        self.optimizer_op = optimizer.minimize(self.cost)

    def residual_block(self, input_tensor, size, rate, dim):
        with tf.variable_scope('fliter%d'%rate):
            conv_filter = self.aconv1d_layer(input_tensor, size=size, rate=rate, activation='tanh')
        with tf.variable_scope('gate%d'%rate):
            conv_gate = self.aconv1d_layer(input_tensor, size=size, rate=rate, activation='sigmoid')
        with tf.variable_scope('conv%d'%rate):
            out = conv_filter * conv_gate
            out = self.conv1d_layer(out, size=1, dim=dim)
        return out + input_tensor, out

    def conv1d_layer(self, input_tensor, size=1, dim=128, bias=False, activation='tanh'):
        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel', (size, shape[-1], dim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        if bias:
            b = tf.get_variable('b', [dim], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.conv1d(input_tensor, kernel, stride=1, padding='SAME') + (b if bias else 0)
        if not bias:
            bn = batch_norm()
            out = bn(out)
            out = self.batch_norm_wrapper(out)

        out = self.activation_wrapper(out, activation)
            
        self.conv1d_index += 1
        return out

    def aconv1d_layer(self, input_tensor, size=7, rate=2, bias=False, activation='tanh'):

        shape = input_tensor.get_shape().as_list()
        kernel = tf.get_variable('kernel',(1, size, shape[-1], shape[-1]), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        if bias:
            b = tf.get_variable('b', [shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0))
        out = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, dim=1), kernel, rate=rate, padding='SAME')
        out = tf.squeeze(out, [1])
        if not bias:
            bn = batch_norm()
            out = bn(out)
            out = self.batch_norm_wrapper(out)

        out = self.activation_wrapper(out, activation)
            
        self.aconv1d_index += 1
        return out

    def batch_norm_wrapper(self, inputs, decay=0.999):
        epsilon = 1e-3
        shape = inputs.get_shape().as_list()    

        beta = tf.get_variable('beta', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
        gamma = tf.get_variable('gamma', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
        pop_mean = tf.get_variable('mean', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(0))
        pop_var = tf.get_variable('variance', shape[-1], dtype=tf.float32, initializer=tf.constant_initializer(1))
        if self.is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, axes=list(range(len(shape)-1)))
            train_mean = tf.assign(pop_mean, pop_mean*decay+batch_mean*(1-decay))
            train_var =tf.assign(pop_var, pop_var*decay+batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)

    def activation_wrapper(self, inputs, activation):
        out = inputs

        if activation == 'sigmoid':
            out = tf.nn.sigmoid(out)
        elif activation == 'tanh':
            out = tf.nn.tanh(out)
        elif activation == 'relu':
            out = tf.nn.relu(out)

        return out