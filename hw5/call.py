import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import gensim
from gensim.models.word2vec import Word2Vec
from gensim import corpora,models,similarities
import io
import time
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout,BatchNormalization
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import keras.backend as K
from keras.layers import Input, Embedding, Merge, Flatten, recurrent, RepeatVector, core, Reshape, Lambda, merge, Convolution2D, MaxPooling2D, Activation
from keras.layers import dot,Dot,Add
import re
import sys
filepath='best_model.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, verbose=1),
    ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
]
def train():
	
	train_data=pd.read_csv('train.csv')
	train_data=train_data.drop('TrainDataID',axis=1)
	print(train_data)

	train_data=train_data.values
	np.random.shuffle(train_data)
	data_len=len(train_data)

	userID=train_data[:,0]
	movieID=train_data[:,1]
	movie_B=np.ones(data_len)
	user_B=np.ones(data_len)

	Rating=train_data[:,2]/5

	user=Input(shape=(1,),name='resp')
	movie=Input(shape=(1,),name='prog')
	movie_bias=Input(shape=(1,),name='movie_bias')
	user_bias=Input(shape=(1,),name='user_bias')
	r=Embedding(input_dim=6041,input_length=1,output_dim=16)(user)
	p=Embedding(input_dim=3955,input_length=1,output_dim=16)(movie)
	m_b=Embedding(input_dim=6041,input_length=1,output_dim=1)(movie_bias)
	u_b=Embedding(input_dim=3955,input_length=1,output_dim=1)(user_bias)
	r=Flatten()(r)
	p=Flatten()(p)
	r=Dropout(0.25)(r)
	p=Dropout(0.25)(p)
	m_b=Flatten()(m_b)
	u_b=Flatten()(u_b)
	DOT=dot([r,p],axes=1)
	ADD=Add()([m_b,u_b,DOT])

	model=keras.models.Model(inputs=[user,movie,movie_bias,user_bias],outputs=[ADD])
	model.summary()
	model.compile(optimizer = "adam", loss = 'mse', 
	              metrics =["accuracy"])
	model.fit([userID,movieID,movie_B,user_B], Rating, batch_size=2048, epochs=200,callbacks=callbacks,validation_split=0.05)
	name=str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
	modelname=name+'.h5'
	model.save(name)


def test():
	testdata=sys.argv[1]
	train_data=pd.read_csv(testdata)
	train_data=train_data.drop('TestDataID',axis=1)
	train_data=train_data.values
	data_len=len(train_data)
	userID=train_data[:,0]
	movieID=train_data[:,1]
	movie_B=np.ones(data_len)
	user_B=np.ones(data_len)

	model=load_model('best.h5')
	y_pred_test=model.predict([userID,movieID,movie_B,user_B])	
	y_pred_test=y_pred_test[:,0]
	y_pred_test=y_pred_test*5
	a=(list(range(len(y_pred_test))))
	a=np.array(a)+1

	out={'TestDataID':a,'Rating':y_pred_test}

	output=DataFrame(data=out)
	cols = output.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	output = output[cols]
	print(output)
	#output=output.astype(int)
	predfile=sys.argv[2]
	output.to_csv(predfile,index=None)



test()

