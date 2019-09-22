# keras
import os
import sys
import pickle
import librosa
import numpy as np
from keras.models import Model
import keras
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model
from keras.optimizers import rmsprop, adam, adagrad, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
from keras.layers import *# Input, Dense, merge, Dropout, BatchNormalization, Activation, Conv1D, Lambda

DIR=os.getcwd()
with open(DIR+"/wav/text/train.word.txt", encoding='utf-8') as f:
    texts=f.read().split("\n")

del texts[-1]
texts=[i.split(" ") for i in texts]
all_words=[]
maxlen_char=0
for i in np.arange(0,len(texts)):
    length=0
    for j in texts[i][1:]:
        length+=len(j)
    if maxlen_char<=length:maxlen_char=length
    for j in np.arange(1,len(texts[i])):
        all_words.append(texts[i][j])

tok=Tokenizer(char_level=True)
tok.fit_on_texts(all_words)
char_index=tok.word_index
index_char=dict((char_index[i],i) for i in char_index)
char_vec=np.zeros((10000,maxlen_char),dtype=np.float32)
#char_input=[[] for _ in np.arange(0,len(texts))]
char_length=np.zeros((10000,1),dtype=np.float32)
for i in np.arange(0,len(texts)):
    j=0
    for i1 in texts[i][1:]:
        for ele in i1:
            char_vec[i,j]=char_index[ele]
            j+=1
    char_length[i]=j

"""
mfcc_vec=[[] for _ in np.arange(0,len(texts))]
for i in np.arange(0,len(texts)):
    wav, sr = librosa.load(DIR + "/wav/train/"+texts[i][0].split('_')[0]+'/'+texts[i][0]+".wav", mono=True)
    b = librosa.feature.mfcc(wav, sr)
    mfcc = np.transpose(b, [1, 0])
    mfcc_vec[i]=mfcc
    if i%100==0:print("Completed {}".format(str(i*len(texts)**-1)))

np.save(DIR+"/mfcc_vec",mfcc_vec)
"""
"""
mfcc_vec_origin=np.load(DIR+"/mfcc_vec_origin.npy")
maxlen_mfcc=673
mfcc_vec=np.zeros((10000,maxlen_mfcc,20),dtype=np.float32)
for i in np.arange(0,len(mfcc_vec_origin)):
    for j in np.arange(0,len(mfcc_vec_origin[i])):
        for k,ele in enumerate(mfcc_vec_origin[i][j]):
            mfcc_vec[i,j,k]=ele

np.save(DIR+"/mfcc_vec",mfcc_vec)
"""

mfcc_input=np.load(DIR+"/39-121915-0014-mfcc_vec.npy")
temp = np.zeros([10000, 700, 20])
for ii in range(10000):
    itsp=np.shape(mfcc_input[ii])
    temp[ii,:itsp[0],:itsp[1]]=mfcc_input[ii]
mfcc_input=temp

vec_shape=(np.shape(mfcc_input[1]))
input_tensor=Input(shape=(vec_shape[0],vec_shape[1]))
x=Conv1D(kernel_size=1,filters=192,padding="same")(input_tensor)
x=BatchNormalization(axis=-1)(x)
x=Activation("tanh")(x)

    
def res_block(xx,size,rate,dim=192):
    x_tanh=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(xx)
    x_tanh=BatchNormalization(axis=-1)(x_tanh)
    x_tanh=Activation("tanh")(x_tanh)
    x_sigmoid=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(xx)
    x_sigmoid=BatchNormalization(axis=-1)(x_sigmoid)
    x_sigmoid=Activation("sigmoid")(x_sigmoid)
    out=merge([x_tanh,x_sigmoid],mode="mul")
    out=Conv1D(kernel_size=1,filters=dim,padding="same")(out)
    out=BatchNormalization(axis=-1)(out)
    out=Activation("tanh")(out)
    xx=merge([xx,out],'sum')
    return xx,out

skip=[]
for i in np.arange(0,3):
    for r in [1,2,4,8,16]:
        x,s=res_block(x,size=7,rate=r)
        skip.append(s)

def ctc_lambda_function(args):
    y_true_input, logit, logit_length_input, y_true_length_input=args
    return K.ctc_batch_cost(y_true_input,logit,logit_length_input,y_true_length_input)

skip_tensor=merge([s for s in skip],'sum')
logit=Conv1D(kernel_size=1,filters=192,padding="same")(skip_tensor)
logit=BatchNormalization(axis=-1)(logit)
logit=Activation("tanh")(logit)
logit=Conv1D(kernel_size=1,filters=len(char_index)+1,padding="same",activation="softmax")(logit)
#base_model=Model(inputs=input_tensor,outputs=logit)
logit_length_input=Input(shape=(1,))
y_true_input=Input(shape=(maxlen_char,))
y_true_length_input=Input(shape=(1,))
loss_out=Lambda(ctc_lambda_function,output_shape=(1,),name="ctc")([y_true_input,logit,logit_length_input,y_true_length_input])
model=Model(inputs=[input_tensor,logit_length_input,y_true_input,y_true_length_input],outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer="adam")
#plot_model(model, to_file="model.png", show_shapes=True)
early = EarlyStopping(monitor="loss", mode="min", patience=10)
lr_change = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=0, min_lr=0.000)
checkpoint = ModelCheckpoint(filepath=DIR + "/listen_model.chk",
                              save_best_only=False)
logdir = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
opt=adam(lr=0.0003)
print("yes1")
#model.load_weights(DIR+"/listen_model.chk")
model.fit(x=[mfcc_input,np.ones(10000)*673,char_vec,char_length],y=np.ones(10000),callbacks=[early,lr_change,checkpoint,logdir],
          batch_size=50,epochs=100)
print("yes2")
model.save_weights('model.hdf5')
with open('model.json', 'w') as f:
    f.write(model.to_json())


