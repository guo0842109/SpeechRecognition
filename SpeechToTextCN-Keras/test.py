import os;
import sys;
import pickle;
import librosa;
import numpy as np;
from math import sqrt
print((type(1+2*3.14)))



wav, sr = librosa.load("1.wav", mono=True);

b = librosa.feature.mfcc(wav, sr)
mfcc = np.transpose(b, [1, 0]);
print(np.shape(wav))
print(np.shape(mfcc))
if i%100==0:print("Completed {}".format(str(i*len(texts)**-1)));
