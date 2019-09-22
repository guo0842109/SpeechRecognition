#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
sequence-loss
=====================
交叉熵损失函数辅助函数
简化编程流程
"""

import tensorflow as tf
import numpy as np

v1 = tf.constant([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  [0, 10, 0],
                  [1, 1, 1]], dtype=tf.float32)
v2 = tf.constant([1, 1, 1, 1, 1], dtype=tf.int32)

loss = (tf.contrib
         .legacy_seq2seq
         .sequence_loss_by_example([v1], 
                                   [v2], 
                                   [tf.ones_like(v2, dtype=tf.float32)]))
sess = tf.Session()
print(sess.run(loss))