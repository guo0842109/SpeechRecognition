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

import codecs
import csv
import os
import re

from hyperparams import Hyperparams as hp
import numpy as np


def load_vocab():
    vocab = "EG abcdefghijklmnopqrstuvwxyz'" # E: Empty. ignore G
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char    

def create_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab() 
      
    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.text_file, 'rb', 'utf-8'))
    for row in reader:
        print(row)
        sound_fname, text, duration = row
        sound_file = hp.sound_fpath + "/" + sound_fname + ".wav"
        text = re.sub(r"[^ a-z']", "", text.strip().lower())
         
        if hp.min_len <= len(text) <= hp.max_len:
            texts.append(np.array([char2idx[char] for char in text], np.int32).tostring())
            sound_files.append(sound_file)
             
    return texts, sound_files
     
def load_train_data():
    """We train on the whole data but the last num_samples."""
    texts, sound_files = create_train_data()
    if hp.sanity_check: # We use a single mini-batch for training to overfit it.
        texts, sound_files = texts[:hp.batch_size]*1000, sound_files[:hp.batch_size]*1000
    else:
        texts, sound_files = texts[:-hp.num_samples], sound_files[:-hp.num_samples]
    return texts, sound_files
 
def load_eval_data():
    """We evaluate on the last num_samples."""
    texts, dirs = create_train_data()

    texts = texts[-hp.num_samples:]
    redirs = dirs[-hp.num_samples:]
    X = np.zeros(shape=[32, hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32) # byte to int
        X[i, :len(_text)] = _text
        print(dirs[i], _text)
    print(np.shape(X), np.max(X), np.min(X))
    return X, dirs 
 

