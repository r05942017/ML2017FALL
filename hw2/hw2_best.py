from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dropout, LeakyReLU, PReLU, MaxoutDense
from keras import metrics
import keras.backend as K
import keras
import tensorflow
import h5py
import sys

test_csv=sys.argv[2]
X_test=sys.argv[5]
outputfilename=sys.argv[6]

feature_num=107

model=load_model('val860.h5')
testDATA=pd.read_csv(X_test)
testDATA=testDATA.values


extra_data=pd.read_csv(test_csv)

buf=np.zeros((len(testDATA),feature_num))
buf[:,:feature_num-1]=testDATA
buf[:,feature_num-1]=extra_data.values[:,4]
testDATA=buf
for i in range(2):
	testDATA[:,i]=(testDATA[:,i]-np.mean(testDATA[:,i]))/np.std(testDATA[:,i])
for i in range(3,6,1):
	testDATA[:,i]=(testDATA[:,i]-np.mean(testDATA[:,i]))/np.std(testDATA[:,i])
testDATA[:,feature_num-1]=(testDATA[:,feature_num-1]-np.mean(testDATA[:,feature_num-1]))/np.std(testDATA[:,feature_num-1])


y_pred_test=model.predict(testDATA)
y_pred_test[y_pred_test>=0.5]=1
y_pred_test[y_pred_test<0.5]=0
y_pred_test=y_pred_test[:,0]

a=(list(range(len(y_pred_test))))
a=np.array(a)+1
out={'id':a,'label':y_pred_test}

output=DataFrame(data=out)
output=output.astype(int)

output.to_csv(outputfilename,index=None)