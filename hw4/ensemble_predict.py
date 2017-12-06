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
import _pickle as pk
import sys

MAX_LEN = 40
TIME_STEP = MAX_LEN
INPUT_SIZE = 200000
INPUT_DIM = 125
BATCH_SIZE = 256
EPOCHS = 10
OUTPUT_SIZE = 1		
TEST_DATA_SIZE = 200000
X_TRAIN_PATH = "./preprocessed_data/texts_train_125.npy"
Y_TRAIN_PATH = "./preprocessed_data/label_train.npy"
TESTING_DATA_PATH = sys.argv[1]
PREDICTION_SAVE_PATH = sys.argv[2]
MODEL_SAVE_PATH = "./models/model_02.h5"
MODEL_LOAD_PATH = "./models/model_01.h5"
W2V_MODEL_PATH = "./gensim_model_125"
ENSEMBLE_MODEL_PATH_ONE = ["./model_1.h5","./model_2.hdf5"]
ENSEMBLE_MODEL_PATH_TWO = ["./model_3.hdf5"]
PREDICTION_PATH = ["new_model_epoch_06_vloss_0.44.npy", "model_02_10_125.npy", "cnn_model_epoch_39_vloss_0.43.npy"]
SAMPLE_ID_PATH = "./preprocessed_data/testing_id.npy"

def build_model(x_train_path, y_train_path, model_save, load = False, model_load = None, model = None):
	
	y_train = np.load(Y_TRAIN_PATH)
	x_train = np.load(X_TRAIN_PATH)


	print(x_train.shape)
	#print(y_train)
	
	print("start to train...")
	

	if load:
		model = load_model(model_load)
		model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
		model.save(model_save)
		return model

	else:
		
		print("start to fit...")
		model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
		model.save(model_save)
		return model

def model_2():
	model = Sequential()
	model.add(BatchNormalization(input_shape=(TIME_STEP, INPUT_DIM)))

	model.add(TimeDistributed(Dense(128)))
	model.add(Bidirectional(GRU(64, return_sequences=True)))
	model.add(TimeDistributed(Dense(128)))
	model.add(Bidirectional(GRU(64)))#final
	#model.add(Dropout(0.5))
	model.add(BatchNormalization())
	
	model.add(Dense(64))
	model.add(Dense(32))
	model.add(Dense(16))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model






def predict(model, sample_id, x_predict, prediction_path):
	print("start to predict")
	y_predict = model.predict(x_predict, verbose = 1)
	predict_label = []
	
	print("data conversting")
	#print(y_predict)
	for prob in y_predict:
		if (prob >= 0.5):
			predict_label.append(1)
		else:
			predict_label.append(0)


	df_output = pd.DataFrame(predict_label, columns = ["label"], index = sample_id)
	df_output.index.name = "id"
	df_output.to_csv(prediction_path, index_label = 'id')

def load_testing_data(file_path):

    with open(file_path, 'r') as f:
        f.readline()

        sentence = []
        sample_id = []
        for i, line in enumerate(f):
        	data = line.split(',', 1)
        	sentence.append(text_to_word_sequence(data[1]))
        	sample_id.append(data[0])



        return np.array(sample_id), sentence

def process_nolabel_data(file_path):

	with open(file_path, 'r') as f:
		sentences = []

		for i, line in enumerate(f):
			sentences.append(text_to_word_sequence(line))

		nb_data = int(len(sentences) / 7)


		for i in range(6):
			tmp_data = sentences[i*nb_data:(i+1)*nb_data]
			tmp_data = dp.word_to_vec(tmp_data, W2V_MODEL_PATH)
			tmp_data = pad_sequences(tmp_data, maxlen=MAX_LEN, padding='post', dtype='float32')
			nolabel_data_path = "./datas/partition_" + str(i+1)
			print(tmp_data.shape)
			np.save(nolabel_data_path, tmp_data)
			del tmp_data

		tmp_data = sentences[6*nb_data:min(len(sentences), 7*nb_data)]
		tmp_data = dp.word_to_vec(tmp_data, W2V_MODEL_PATH)
		tmp_data = pad_sequences(tmp_data, maxlen=MAX_LEN, padding='post', dtype='float32')
		nolabel_data_path = "./datas/partition_" + str(7)
		print(tmp_data.shape)
		np.save(nolabel_data_path, tmp_data)
		del tmp_data



def self_training(model, unlabel_data, threshold = 0):
	unlabel_preditions = model.predict(unlabel_data)

	suedo_training = []
	for i, prediction in unlabel_preditions:
		if (prediction > threshold):
			tmp_list = []
			tmp_list.append(1)
			tmp_list.append(unlabel_data[i])
			suedo_training.append(tmp_list)

		elif(prediction < (1.0 - threshold)):
			tmp_list = []
			tmp_list.append(0)
			tmp_list.append(unlabel_data[i])
			suedo_training.append(tmp_list)

	return suedo_training


def voting(predictions, sample_id, prediction_save_path):
	predict = np.zeros(TEST_DATA_SIZE)
	for prediction in predictions:
		predict = predict + prediction.reshape(TEST_DATA_SIZE)

	prediction = predict/float(len(predictions))

	predict_label = []
	for prob in prediction:
		if (prob >= 0.5):
			predict_label.append(1)
		else:
			predict_label.append(0)

	df_output = pd.DataFrame(predict_label, columns = ["label"], index = sample_id)
	df_output.index.name = "id"
	df_output.to_csv(prediction_save_path, index_label = 'id')



def main():

	
	sample_id, x_test = load_testing_data(TESTING_DATA_PATH)
	print(type(x_test[0]))
	print("word to vec ...")
	x_test = dp.word_to_vec(x_test, W2V_MODEL_PATH)
	print("start to padding ...")
	x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', dtype='float32')

	#sample_id = np.load("./preprocessed_data/testing_id.npy")[:100]
	#x_test = np.load("./preprocessed_data/testing_data_125.npy")[:100]

	predictions = []
	model1 = load_model(ENSEMBLE_MODEL_PATH_ONE[0])
	predictions.append(model1.predict(x_test))
	print("model1 success")
	model2 = load_model(ENSEMBLE_MODEL_PATH_ONE[1])
	predictions.append(model2.predict(x_test))
	print("model2 success")
	model3 = load_model(ENSEMBLE_MODEL_PATH_TWO[0])
	x_test = x_test.reshape((-1,40,125,1))
	predictions.append(model3.predict(x_test)[:, 1])
	print("model3 success")
	voting(predictions, sample_id, PREDICTION_SAVE_PATH)
	#sample_id = np.load("./preprocessed_data/testing_id.npy")
	#x_test = np.load("./preprocessed_data/testing_data_125_nofilter.npy")
	
	
	

	#build_model("texts_train_125_nofilter.npy", "./preprocessed_data/testing_id.npy", "./models/model_2_nofilter.h5", model = model_2())
	#voting(PREDICTION_PATH, PREDICTION_SAVE_PATH)

main()