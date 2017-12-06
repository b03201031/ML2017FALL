import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import _pickle as pk
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.layers import normalization, Average, Add, Bidirectional, GRU, SimpleRNN, TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization, LSTM
from keras.layers import Dense, Merge, Concatenate, Multiply, Subtract
from keras.optimizers import SGD, Adam, rmsprop, Adadelta
from keras.callbacks import EarlyStopping
import data_preprocessing as dp
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from sklearn.ensemble import AdaBoostClassifier


class DataManager:
	def __init__(self):
		self.data = {}
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
	def add_data(self,name, data_path, with_label=True):
		print ('read data from %s...'%data_path)
		X, Y = [], []
		with open(data_path,'r') as f:
			#f.readline()
			for line in f:
				if with_label :
					#lines = line.strip().split(',', 1)
					lines = line.strip().split(' +++$+++ ')
					X.append(lines[1])
					Y.append(int(lines[0]))
				else:
					X.append(line)

				if with_label:
					self.data[name] = [X,Y]
				else:
					self.data[name] = [X]


    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
	def tokenize(self, vocab_size):
		print ('create new tokenizer')
		self.tokenizer = Tokenizer(num_words=vocab_size)
		for key in self.data:
			print ('tokenizing %s'%key)
			texts = self.data[key][0]
			self.tokenizer.fit_on_texts(texts)
        
    # Save tokenizer to specified path
	def save_tokenizer(self, path):
		print ('save tokenizer to %s'%path)
		pk.dump(self.tokenizer, open(path, 'wb'))
            
    # Load tokenizer from specified path
	def load_tokenizer(self,path):
		print ('Load tokenizer from %s'%path)
		self.tokenizer = pk.load(open(path, 'rb'))

	def to_bow(self):
		for key in self.data:
			print ('Converting %s to tfidf'%key)
			self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')
 
 


def process_data_BOW(train_path, vocab_size):
	print("hello")
	dm = DataManager()

	print("adding data")
	dm.add_data('train_data', train_path, True)
	dm.tokenize(vocab_size)
	print(dm.data['train_data'][0][0])
	dm.to_bow()
	print(dm.data['train_data'][0][0])

	'''
	print ('get Tokenizer...')
	dm.tokenize(vocab_size)
	#dm.save_tokenizer("BOW_tokenizer.pkl")
	dm.load_tokenizer("BOW_tokenizer.pkl")
	dm.to_bow()
	
	bow_text = np.asarray(dm.data['train_data'][0])
	print(bow_text.shape)
	'''
process_data_BOW("./datas/training_label.txt", 8000)

'''
print("loading...")
x_train = np.load("./preprocessed_data/training_label_bow_2.npy")
y_train = np.load("label_train.npy")
y_train = y_train[int(len(y_train)/2.):]
print("done")

#print(x_train)
#print(y_train)

model = Sequential()
model.add(Dense(256, input_shape=(8000,)))
model.add(Dense(126))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = load_model("./models/BOW_model.h5")
print(x_train.shape)
#model.fit(x_train, y_train, batch_size = 256, epochs= 20, validation_split=0.1)
#model.save("./models/BOW_model.h5")
'''