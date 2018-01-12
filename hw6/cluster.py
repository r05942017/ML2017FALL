import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense,Bidirectional,Flatten,Subtract,Lambda,Masking,dot,maximum,PReLU,LeakyReLU,Dropout,BatchNormalization,Activation
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
import sys

imagepath=sys.argv[1]
test_case=sys.argv[2]
prediction_file_path=sys.argv[3]

filepath='best_model.h5'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=50, verbose=1),
    ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, verbose=1,save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
]
a=np.load(imagepath)
print(a.shape)
a=a.astype('float32')
a=a/255

input_img = Input(shape=(784,))
encoded = Dense(512)(input_img)
#encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = Dense(256)(input_img)
#encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)	
encoded = Dense(128)(encoded)
#encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = Dense(64)(encoded)
#encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)
encoded = Dense(32)(encoded)
#encoded = BatchNormalization()(encoded)
encoded = Activation('relu')(encoded)




decoded = Dense(64)(encoded)
#decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = Dense(128)(decoded)
decoded = Activation('relu')(decoded)
decoded = Dense(256)(decoded)
#decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = Dense(512)(decoded)
#decoded = BatchNormalization()(decoded)
decoded = Activation('relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
def train():
	autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
	autoencoder.summary()
	autoencoder.fit(a, a,epochs=1000,batch_size=256,callbacks=callbacks,validation_split=0.05)
	autoencoder.save('h6.h5')



def test():

	autoencoder.load_weights('81749.h5')
	compress=Model(input_img,encoded)
	compress_a=compress.predict(a)
	print(compress_a.shape)
	pred_model=KMeans(n_clusters=2).fit(compress_a)
	print(pred_model.labels_[0])


	test=pd.read_csv(test_case)
	print(test.values)
	test=test.values
	label=np.zeros(len(test),dtype=int)
	for i in range(len(test)):
		if int(pred_model.labels_[int(test[i,1])])==int(pred_model.labels_[int(test[i,2])]):
			label[i]=1
		if i%100000==0:
			print(i,label[i])

	lis=(list(range(len(label))))

	lis=np.array(lis)

	#y_pred_test=y_pred_test.reshape(200000)
	labelname=np.array(['ID','Ans'])
	output=np.zeros((len(label),2),dtype=int)
	output[:,0]=lis
	output[:,1]=label
	DAT =  np.row_stack((labelname, output))

	np.savetxt(prediction_file_path, DAT, delimiter=',',fmt="%s")

test()