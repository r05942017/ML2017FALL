import warnings 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
import pickle
import numpy as np
import pandas as pd
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM, Input, Merge
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import h5py


### Read Raw Data Start ###
# test audio
with open(sys.argv[1],'rb') as audio_data:   # (2000,246,39)
	test_audio_raw = pickle.load(audio_data)
# test caption
test_caption_raw = pd.read_csv(sys.argv[2], header=None) # (2000,4)
test_caption_raw = test_caption_raw.values
### Read Raw Data End ###


### Parameter ###
# test number
test_num = 2000
# audio
audio_max_length = 246
audio_vector_length = 39
# caption
chinese_embedding_file = "chinese_embedding.model"
caption_max_length = 13
caption_vector_length = 300
choices = 4
# Seq2Seq
latent_dim = 256
Drop = 0.6
### Parameter ###




### Encoder Decoder Data Processing Start ###
# Encoder Input Data
encoder_input_data = np.zeros((test_num,audio_max_length,audio_vector_length))
for i in range(test_num):
	for j in range(len(test_audio_raw[i])):
		### Normalization ###
		AVG = np.mean(test_audio_raw[i][j])
		STD = np.std(test_audio_raw[i][j])
		encoder_input_data[i][j] = (test_audio_raw[i][j]-AVG)/STD
		### Normalization ###
total_encoder_input_data = np.repeat(encoder_input_data,choices,axis=0)
del encoder_input_data
print('total_encoder_input_data.shape',total_encoder_input_data.shape)
##################################################################################
# decoder caption embedding & train caption data
caption_word_model = KeyedVectors.load(chinese_embedding_file)
# decoder input data (train_num,caption_max_length,caption_vector_length)
test_caption_raw = np.reshape(test_caption_raw,(choices*test_num,1))
total_decoder_input_data = np.zeros((choices*test_num,caption_max_length,caption_vector_length))
sparse_data = 0
for i in range(choices*test_num):
	sentence_split_data = test_caption_raw[i][0].split()
	for j in range(len(sentence_split_data)):
		if sentence_split_data[j] in caption_word_model.wv.vocab:
			### Normalization ###
			AVG = np.mean(caption_word_model[sentence_split_data[j]])
			STD = np.std(caption_word_model[sentence_split_data[j]])
			total_decoder_input_data[i][j] = (caption_word_model[sentence_split_data[j]]-AVG)/STD
		else:
			sparse_data += 1
print('total_decoder_input_data.shape:',total_decoder_input_data.shape)



### Match Model Start ###
# Encoder Model
encoder_model = Sequential()
encoder_model.add(LSTM(latent_dim, input_shape=(audio_max_length,audio_vector_length), return_sequences=False, go_backwards=True, unroll=True, implementation=2))
encoder_model.add(Dense(128))
encoder_model.add(BatchNormalization())
encoder_model.add(Activation('linear'))
#encoder_model.add(Dropout(Drop))
encoder_model.summary()


# Decoder Model, using `encoder_states` as decoder initial state
decoder_model = Sequential()
decoder_model.add(LSTM(latent_dim, input_shape=(caption_max_length,caption_vector_length), return_sequences=False, go_backwards=True, unroll=True, implementation=2))
decoder_model.add(Dense(128))
decoder_model.add(BatchNormalization())
decoder_model.add(Activation('linear'))
#decoder_model.add(Dropout(Drop))
decoder_model.summary()


model = Sequential()
model.add(Merge([encoder_model,decoder_model], mode='dot'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
### Match Model Stop ###

model.load_weights('imp-14-0.6484-48.7_batch.h5')
predict1 = model.predict([total_encoder_input_data,total_decoder_input_data])
model.load_weights('imp-13-0.7044_Normal.h5')
predict2 = model.predict([total_encoder_input_data,total_decoder_input_data])
predict = (predict1+predict2)/2

final_ans = []
for i in range(0,choices*test_num,choices):
	right_score = []
	for j in range(choices):
		right_score.append(predict[i+j][0])
	right_score_index = np.argmax(right_score)
	final_ans.append(right_score_index)
print('len(final_ans):',len(final_ans))

# Export
predict_out = []
predict_out.append("id,answer")
for i in range(test_num):
	predict_out.append(str(i+1) + "," + str(final_ans[i]))
f = open(sys.argv[3],'w')
for i in range(test_num+1):
	f.write(predict_out[i] + "\n")
f.close()
