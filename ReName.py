#-*- coding:utf-8 -*-
#用于变量改名使其适用于1.2以上版本
#cangye@hotmail.com
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

new_checkpoint_vars = {}
reader = pywrap_tensorflow.NewCheckpointReader(os.getcwd()+"/model/ckpt-0")

for old_name in reader.get_variable_to_shape_map():
    print(old_name)
    if "/lstm_cell/biases" in old_name:
        new_name = old_name.replace("/lstm_cell/biases", "/lstm_cell/bias")
    elif "/lstm_cell/weights" in old_name:
        new_name = old_name.replace("/lstm_cell/weights", "/lstm_cell/kernel")
    else:
        new_name = old_name
    new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))
"""
init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
    sess.run(init)
    saver.save(sess, os.getcwd() + "/seq2seq/model/chatbot_seq2seq.ckpt-10888")
    print("checkpoint file rename successful... ")
"""