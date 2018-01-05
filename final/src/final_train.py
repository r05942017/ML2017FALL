import warnings 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
from sklearn.utils import shuffle
import pickle
import numpy as np
import pandas as pd
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, LSTM, Input, Merge
from keras.callbacks import ModelCheckpoint
import h5py


### Read Raw Data Start ###
# train audio
with open(sys.argv[1],'rb') as audio_data:   # (45036,246,39)
	train_audio_raw = pickle.load(audio_data)
# train caption
train_caption_raw = pd.read_csv(sys.argv[2], header=None) # (45036,1)
train_caption_raw = train_caption_raw.values
### Read Raw Data End ###


### Parameter ###
# train number
train_num = 45036 # 45036
# audio
audio_max_length = 246
audio_vector_length = 39
# caption
chinese_embedding_file = "chinese_embedding.model"
caption_max_length = 13
caption_vector_length = 300
# Seq2Seq
latent_dim = 256
Drop = 0.3
### Parameter ###



### Encoder Decoder Data Processing Start ###
# Encoder Input Data
encoder_input_data = np.zeros((train_num,audio_max_length,audio_vector_length))
for i in range(train_num):
	for j in range(len(train_audio_raw[i])):
		# Normalization
		AVG = np.mean(train_audio_raw[i][j])
		STD = np.std(train_audio_raw[i][j])
		encoder_input_data[i][j] = (train_audio_raw[i][j]-AVG)/STD
		# Normalization
total_encoder_input_data = np.vstack((encoder_input_data,encoder_input_data))
del encoder_input_data
print('total_encoder_input_data.shape:',total_encoder_input_data.shape)
##############################################################################################
# decoder caption embedding & train caption data
caption_word_model = KeyedVectors.load(chinese_embedding_file)
# decoder input data right (train_num,caption_max_length,caption_vector_length)
decoder_input_data_right = np.zeros((train_num,caption_max_length,caption_vector_length))
sparse_data = 0 # total = 26
for i in range(train_num):
	sentence_split_data = train_caption_raw[i][0].split()
	for j in range(len(sentence_split_data)):
		if sentence_split_data[j] in caption_word_model.wv.vocab:
			# Normalization
			AVG = np.mean(caption_word_model[sentence_split_data[j]])
			STD = np.std(caption_word_model[sentence_split_data[j]])
			decoder_input_data_right[i][j] = (caption_word_model[sentence_split_data[j]]-AVG)/STD
			# Normalization
		else:
			sparse_data += 1
# decoder input data wrong (train_num,caption_max_length,caption_vector_length)
decoder_input_data_wrong = np.zeros((train_num,caption_max_length,caption_vector_length))
for i in range(train_num):
	random_index = np.random.choice(train_num,1)
	while random_index is i:
		random_index = np.random.choice(train_num,1)
	sentence_split_data = train_caption_raw[random_index[0]][0].split()
	for j in range(len(sentence_split_data)):
		if sentence_split_data[j] in caption_word_model.wv.vocab:
			# Normalization
			AVG = np.mean(caption_word_model[sentence_split_data[j]])
			STD = np.std(caption_word_model[sentence_split_data[j]])
			decoder_input_data_wrong[i][j] = (caption_word_model[sentence_split_data[j]]-AVG)/STD
			# Normalization
		else:
			sparse_data += 1
total_decoder_input_data = np.vstack((decoder_input_data_right,decoder_input_data_wrong))
del decoder_input_data_right, decoder_input_data_wrong
print('total_decoder_input_data.shape:',total_decoder_input_data.shape)
#############################################################################################
# label (right,wrong)
label_right = np.ones((train_num,1))
label_wrong = np.zeros((train_num,1))
total_label_data = np.vstack((label_right,label_wrong))
del label_right, label_wrong
print('total_label_data.shape:',total_label_data.shape)
#############################################################################################
# Shuffle Simultaneously
total_encoder_input_data, total_decoder_input_data, total_label_data = shuffle(total_encoder_input_data, total_decoder_input_data, total_label_data, random_state=None)
#total_encoder_input_data, total_decoder_input_data, total_label_data = shuffle(total_encoder_input_data, total_decoder_input_data, total_label_data, random_state=None)
#total_encoder_input_data, total_decoder_input_data, total_label_data = shuffle(total_encoder_input_data, total_decoder_input_data, total_label_data, random_state=None)
print('shuffle_end')
#############################################################################################



### Match Model Start ###
# Encoder Model
encoder_model = Sequential()
encoder_model.add(LSTM(latent_dim, input_shape=(audio_max_length,audio_vector_length), return_sequences=False, go_backwards=True, unroll=True, implementation=2))
#encoder_model.add(Bidirectional(LSTM(latent_dim, return_sequences=False, implementation=2), input_shape=(audio_max_length,audio_vector_length), merge_mode='sum'))
encoder_model.add(Dense(128))
encoder_model.add(BatchNormalization())
encoder_model.add(Activation('linear'))
#encoder_model.add(Dropout(Drop))
encoder_model.summary()


# Decoder Model, using `encoder_states` as decoder initial state
decoder_model = Sequential()
decoder_model.add(LSTM(latent_dim, input_shape=(caption_max_length,caption_vector_length), return_sequences=False, go_backwards=True, unroll=True, implementation=2))
#decoder_model.add(Bidirectional(LSTM(latent_dim, return_sequences=False, implementation=2), input_shape=(audio_max_length,audio_vector_length), merge_mode='sum'))
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


# Run training
filepath="imp-{epoch:02d}-{val_acc:.4f}_Normal.h5"
checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1, mode='max', period=1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([total_encoder_input_data, total_decoder_input_data], total_label_data, batch_size=128, epochs=50, validation_split=0.1, callbacks=[checkpointer])
model.save_weights('final_epoch.h5')



