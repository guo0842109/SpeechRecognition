#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
ctc-loss
=====================
非定长序列相似度对比
类似于动态规划算法
"""
print(__doc__)

import tensorflow as tf
import numpy as np

v1 = tf.Variable([[1, 1, 2]], dtype=tf.int32)
indices = tf.where(tf.not_equal(tf.cast(v1, tf.float32), 0.))
label = tf.SparseTensor(indices=indices, 
                values=tf.gather_nd(v1, indices), 
                dense_shape=tf.cast(tf.shape(v1), tf.int64))
v2 = tf.Variable([[[0.1, 0.8, 0.1, 0.1],
                   [0.1, 0.8, 0.1, 0.1],
                   [0.1, 0.1, 0.1, 0.8],
                   [0.1, 0.1, 0.8, 0.1],
                   [0.1, 0.1, 0.1, 0.8],
                   [0.1, 0.1, 0.1, 0.8]]])
#print(tf.shape(v2))
ctc = tf.nn.ctc_loss(label, v2, sequence_length=[4], time_major=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(ctc))
