import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import gensim
import io
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout,BatchNormalization
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import sys

training_label=sys.argv[1]
training_nolabel=sys.argv[2]


filepath='best_model.h5'
callbacks = [
    EarlyStopping(monitor='val_acc', patience=30, verbose=1),
    ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
]
#------------------------------------------------------build word embeding


print('build word embeding')
data=pd.read_csv(training_label,sep='\n',header=None)
data=Series(data[0])
data=data.str.split(' ',n=2,expand=True)
data=Series(data[2])

nolabel_data=pd.read_csv(training_nolabel,sep='\n',header=None)
nolabel_data=Series(nolabel_data[0])
total_data = pd.concat([data, nolabel_data], ignore_index=True)
total_data=total_data.tolist()

for i in range(len(total_data)):
	total_data[i]= total_data[i].split()


dict_model = Word2Vec(total_data,min_count=1)
dict_model.save("dict.model")

#------------------------------------------------------build word embeding


model_2 = Word2Vec.load("dict.model")
data=pd.read_csv(training_label,sep='\n',header=None)
data=Series(data[0])
data=data.str.split(' ',n=2,expand=True)
label=data[0].values
data=Series(data[2])

#test=data.str.split(' ',expand=True)
#print(test)

data=data.tolist()
#print(model_2.most_similar(['taiwan']) )
for i in range(len(data)):
	data[i]= data[i].split()

#a=model_2[data[8]]
stringlen=50
input_data=np.zeros((int(len(data)),stringlen,100))
for i in range(len(data)):
	buf=model_2[data[i]]
	if len(buf)>=stringlen:
		input_data[i,:,:]=buf[:stringlen,:]
	else:
		input_data[i,:len(buf),:]=buf


print('start train')
model = Sequential()
model.add(LSTM(256,input_shape=(stringlen, 100),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('training')
model.fit(input_data, label, batch_size=512, epochs=30,callbacks=callbacks,validation_split=0.05)

model.save('1127.h5')


