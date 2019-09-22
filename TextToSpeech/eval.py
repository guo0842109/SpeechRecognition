# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
语音生成网络模型
参考文献reference/文件夹下
关键点：
梅尔谱
STFT谱
Attention

Acknowledgment：
kyubyong park. kbpark.linguist@gmail.com. 
Arignal Source：
https://www.github.com/kyubyong/tacotron
'''

from __future__ import print_function

import codecs
import copy
import os

import librosa
from scipy.io.wavfile import write

from hyperparams import Hyperparams as hp
import numpy as np
from prepro import *
import tensorflow as tf
from train import Graph
from utils import *


def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, _ = load_eval_data() # texts
    #print(_)
    char2idx, idx2char = load_vocab()
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.model))
            print("Restored!")
             
            # Get model
            mname = open(hp.model + '/checkpoint', 'r').read().split('"')[1] # model name

            timesteps = 100  # Adjust this number as you want
            outputs1 = np.zeros((hp.num_samples, timesteps, hp.n_mels * hp.r), np.float32)  # hp.n_mels*hp.r
            X
            for j in range(timesteps):
                _outputs1 = sess.run(g.outputs1, {g.x: X, g.y: outputs1})
                outputs1[:, j, :] = _outputs1[:, j, :]
            outputs2 = sess.run(g.outputs2, {g.outputs1: outputs1})

    # Generate wav files
    if not os.path.exists(hp.outputdir): os.mkdir(hp.outputdir) 
    with codecs.open(hp.outputdir + '/text.txt', 'w', 'utf-8') as fout:
        for i, (x, s) in enumerate(zip(X, outputs2)):
            if np.sum(x) == 0:
                continue
            # write text
            fout.write(str(i) + "\t" + "".join(idx2char[idx] for idx in np.fromstring(x, np.int32) if idx != 0) + "\n")
            
            s = restore_shape(s, hp.win_length//hp.hop_length, hp.r)
                         
            # generate wav files
            if hp.use_log_magnitude:
                audio = spectrogram2wav(np.power(np.e, s)**hp.power)
            else:
                s = np.where(s < 0, 0, s)
                audio = spectrogram2wav(s**hp.power)
            write(hp.outputdir + "/{}_{}.wav".format(mname, i), hp.sr, audio)
if __name__ == '__main__':
    eval()
    print("Done")
    
    
