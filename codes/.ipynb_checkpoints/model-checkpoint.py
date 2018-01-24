#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:59:03 2018

@author: bob
"""

import keras as k
import keras.backend as kb
from keras.layers import Conv2D,Flatten,Dense,Input,MaxPool2D,Dropout,Multiply,merge
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt

_trdir = "/home/bob/Desktop/code/GalaxyZoo/Data/V4-96p/train/"
_tsdir = "/home/bob/Desktop/code/GalaxyZoo/Data/V4-96p/test/"

X_train = np.load(_trdir+"x.npy")
y_train = np.load(_trdir+"Y.npy")
"""
X_train = X_train.reshape(y_train.shape[0],424,424,3)
X_train_g = np.sum(X_train,axis=3)/3
X_train_g = X_train_g.reshape(y_train.shape[0],424,424,1)
"""
print("train data loaded\n"+str(X_train.shape))

X_test = np.load(_tsdir+"x.npy")
y_test = np.load(_tsdir+"Y.npy")
"""
X_test = X_test.reshape(y_test.shape[0],424,424,3)
X_test_g = np.sum(X_test,axis=3)/3
X_test_g = X_test_g.reshape(y_test.shape[0],424,424,1)
"""

print("\n\nData loaded\n\n")
#%%
X_train_g = (X_train[:,110:314,110:314]/np.max(X_train))**1.5
X_test_g = X_test[:,110:314,110:314]/np.max(X_test)**1.5

plt.imshow(X_train_g[6].reshape(204,204))
plt.show()
#%%
class LossHistory(k.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.ev = []

    def on_epoch_end(self,epoch, batch, logs={}):
        if epoch % 1 == 0:
            e = self.model.evaluate(X_test, y_test,verbose=0)
            #print("\n{}".format(e))
            self.ev.append(e)

def loss(y_true, y_pred):
        
        return k.mean(k.square(y_pred - y_true) - k.square(y_true), axis=-1)
    return loss
#%%
inputs = Input(shape=(96,96,1))

c1 = Conv2D(4,(2,2),strides=(2,2),padding='same',activation='relu')(inputs)
c1 = Conv2D(16,(2,2),strides=(1,1),padding='same',activation='relu')(c1)
c1 = Conv2D(32,(2,2),strides=(2,2),padding='same',activation='relu')(c1)


f1 = Flatten()(c1)

#con = merge([f1,f2],mode='sum',concat_axis=-1)

d = Dense(32,activation='relu')(f1)
d = Dropout(0.25)(d)
d = Dense(8,activation='relu')(d)
d = Dropout(0.75)(d)

dn3 = Dense(2,activation='sigmoid')(d)

model = Model(inputs=inputs,outputs=dn3)
model.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['acc'])
model.summary()
#%%
h = LossHistory()
history = model.fit(X_train,y_train,epochs=128,batch_size=2,callbacks=[h],verbose=1)

ev = np.array(h.ev) 
ev = ev[:,1]

plt.figure(figsize=(8, 5), dpi=80)
plt.plot(history.history['acc'])
plt.plot(ev)
#plt.plot(history.history['loss'],"-")
plt.legend(["train","test","train loss"])
plt.show()
