from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU, MaxoutDense,ZeroPadding2D
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import metrics,regularizers
import keras.backend as K
import keras
import tensorflow
import h5py
import sys
testDATA=sys.argv[1]
filepath='best_model.h5'
callbacks = [
    EarlyStopping(monitor='val_acc', patience=30, verbose=1),
    ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
]

DATA = np.genfromtxt(testDATA,delimiter=',',dtype=None)
DATA=DATA[1:,:]

data=np.zeros((len(DATA),48,48,1))
for i in range(len(DATA)):
	data[i,:,:,0]=np.fromstring(DATA[i,1], dtype=int, sep=' ').reshape((48,48))
	if np.std(data[i,:,:,0]) !=0:
		data[i,:,:,0]=data[i,:,:,0]/np.std(data[i,:,:,0])


label=np.zeros((len(DATA),7))
label_buf=DATA[:,0].astype(int)
for i in range(len(DATA)):
	label[i,label_buf[i]]=1


DROPOUT=0.3

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model.add(Conv2D(32,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(DROPOUT))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(DROPOUT))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Dropout(DROPOUT))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))

model.add(Dense(7))
model.add(Activation("softmax"))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data,label,epochs=1,callbacks=callbacks,batch_size=128,validation_split=0.1)


model.save('model.h5')
