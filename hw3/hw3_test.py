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


model_name='hw3_model.h5'
testDATA=sys.argv[1]
pred_name=sys.argv[2]



model=load_model(model_name)

DATA = np.genfromtxt(testDATA,delimiter=',',dtype=None)
DATA=DATA[1:,:]

data=np.zeros((len(DATA),48,48,1))
for i in range(len(DATA)):
	data[i,:,:,0]=np.fromstring(DATA[i,1], dtype=int, sep=' ').reshape((48,48))
	if np.std(data[i,:,:,0])!=0:
		data[i,:,:,0]=data[i,:,:,0]/np.std(data[i,:,:,0])



y_pred_test=model.predict(data)
buf=np.zeros(len(DATA))
for i in range(len(DATA)):
	buf[i]=int(np.array(np.where(y_pred_test==np.max(y_pred_test[i,:])))[1,0])
y_pred_test=buf

a=(list(range(len(y_pred_test))))
a=np.array(a)


labelname=np.array(['id','label'])
output=np.zeros((len(y_pred_test),2),dtype=int)
output[:,0]=a
output[:,1]=y_pred_test
DAT =  np.row_stack((labelname, output))

np.savetxt(pred_name, DAT, delimiter=',',fmt="%s")
