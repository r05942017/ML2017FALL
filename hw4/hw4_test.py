import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import gensim
import io
import time
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout,BatchNormalization
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import sys

testing_data=sys.argv[1]
PRED_model='1127.h5'
PRED_name=sys.argv[2]
print('load dict')
dict_model = Word2Vec.load("dict.model")
pred_model=load_model(PRED_model)
pred_model.summary()
print('load data')
test_data=pd.read_csv(testing_data,sep='\n')
print('transfer data')
test_data=Series(test_data['id,text'])
test_data=test_data.str.split(',',n=1,expand=True)
test_data=Series(test_data[1])
test_data=test_data.tolist()
for i in range(len(test_data)):
	test_data[i]= test_data[i].split()

stringlen=50
input_data=np.zeros((int(len(test_data)),stringlen,100))
for i in range(len(test_data)):
	buf=np.zeros((len(test_data[i]),100))
	for j in range(len(test_data[i])):
		try:
			buf[j]=dict_model[test_data[i][j]]
		except KeyError :
			pass


	if len(buf)>=stringlen:
		input_data[i,:,:]=buf[:stringlen,:]
	else:
		input_data[i,:len(buf),:]=buf
print('predict')


y_pred_test=pred_model.predict(input_data)


y_pred_test[y_pred_test>=0.5]=1
y_pred_test[y_pred_test<0.5]=0
a=(list(range(len(y_pred_test))))
a=np.array(a)


labelname=np.array(['id','label'])
output=np.zeros((len(y_pred_test),2),dtype=int)
output[:,0]=a
output[:,1]=y_pred_test[:,0]
DAT =  np.row_stack((labelname, output))

np.savetxt(PRED_name, DAT, delimiter=',',fmt="%s")
