import argparse
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import pickle as pk
from sklearn.decomposition import NMF
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.layers import normalization, Average, Add, Bidirectional, GRU, SimpleRNN, TimeDistributed, Embedding, Dot, Lambda
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization, LSTM
from keras.layers import Dense, Merge, Concatenate, Multiply, Subtract, Masking
from keras.optimizers import SGD, Adam, rmsprop, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras import backend as K
import random
import sys

parser = argparse.ArgumentParser(description='Rating')
#parser.add_argument('model')
parser.add_argument('--action', choices=['train', 'test'])
parser.add_argument('--training_path')
parser.add_argument('--testing_path')
parser.add_argument('--val_ratio', type=int, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--model_save_path')
parser.add_argument('--model_folder')
parser.add_argument('--embeding_dim', default=20, type=int)
parser.add_argument('--model_load_path')
parser.add_argument('--prediction_save_path')
parser.add_argument('--train_option')
parser.add_argument('--regularizers_scale', default = 0, type=float)


args = parser.parse_args()




MAX_USER_ID = 6040
MAX_MOVIE_ID = 3952


def testing(x_user, x_movie, one_matrix, model_path):
	#all_user_predicted_ratings = np.load(args.predicted_ratings_path)
	print("load model from", model_path)
	model = load_model(model_path, custom_objects={'getRMSE': getRMSE})
	print(model.summary())
	y_pre = model.predict([x_user, x_movie, one_matrix], verbose = True) #start from 0
	return y_pre

def RMSE(y_pre, y_true):
	return np.sqrt(((y_pre - y_true)**2).mean())
	
def train_model():
	train = [i.strip('\n').split(',') for i in open(args.training_path, 'r').readlines()]
	train = (np.asarray(train[1:]).astype(int))[:, 1:]
	np.random.shuffle(train)

	x_user = train[:, 0].astype(float)
	x_movie = train[:, 1].astype(float)
	one_matrix = np.ones(len(x_movie))
	print(x_user.shape)
	print(x_movie.shape)
	print(one_matrix.shape)
	
	y_rating = train[:, 2].astype(float)
	y_rating = y_rating 
	print(y_rating)

	if args.train_option == "build":
		k = 128
		input_user = Input(shape=(1,), dtype = "int32")
		embeding_user = Embedding(MAX_USER_ID, args.embeding_dim, embeddings_regularizer = l2(args.regularizers_scale), embeddings_initializer = RandomNormal())(input_user)
		embeding_user = Flatten()(embeding_user)
		
		input_movie = Input(shape=(1,), dtype = "int32")
		embeding_movie = Embedding(MAX_MOVIE_ID, args.embeding_dim, embeddings_regularizer = l2(args.regularizers_scale), embeddings_initializer = RandomNormal())(input_movie)
		embeding_movie = Flatten()(embeding_movie)

		embeding_user_bias = Embedding(MAX_USER_ID, output_dim = 1, input_length = 1, embeddings_regularizer = l2(args.regularizers_scale), embeddings_initializer = RandomNormal())(input_user)
		embeding_user_bias = Flatten()(embeding_user_bias)
		embeding_movie_bias = Embedding(MAX_MOVIE_ID, output_dim = 1, input_length = 1, embeddings_regularizer = l2(args.regularizers_scale), embeddings_initializer = RandomNormal())(input_movie)
		embeding_movie_bias = Flatten()(embeding_movie_bias)

		bias_user = Dense(1)(embeding_user_bias)
		bias_movie = Dense(1)(embeding_movie_bias)

		dot_layer = Dot(axes=-1)([embeding_user, embeding_movie])

		input_one = Input(shape = (1, ))
		bias_rating = Dense(1, activation = "linear", use_bias = False,
			kernel_regularizer = l2(0.0), kernel_initializer = RandomNormal())(input_one)

		
		dotBiasSumLayer = Lambda(getDotBiasSum, output_shape = getDotBiasSumShape)\
		([dot_layer, bias_rating, bias_movie, bias_user])


		opt = Adam()
		model = Model(inputs=[input_user, input_movie, input_one], outputs=dotBiasSumLayer)
		model.compile(loss='mean_squared_error', optimizer=opt, metrics = [getRMSE])
	
	if args.train_option == "load":
		model = load_model(args.model_folder + args.model_load_path)
	saver = ModelCheckpoint(args.model_folder+'./keras_model_epoch_{epoch:02d}_vloss_{val_loss:.2f}.hdf5')
	model.fit([x_user, x_movie, one_matrix], y_rating, batch_size = args.batch_size, epochs=args.epochs, validation_split=0.1,callbacks=[saver])


def getRMSE(labelMatrix, predictionMatrix):
	return K.sqrt(K.mean(K.square(labelMatrix - predictionMatrix)));


def getDotBiasSum(parameterMatrixList):
	dotMatrix, userBiasMatrix, itemBiasMatrix, ratingBiasMatrix = parameterMatrixList;
	return dotMatrix + userBiasMatrix + itemBiasMatrix + ratingBiasMatrix;

def getDotBiasSumShape(shapeVectorList):
	dotShapeVector, userBiasShapeVector, itemBiasShapeVector, ratingBiasShapeVector = shapeVectorList;
	return userBiasShapeVector[0], 1;


def main():
	if args.action == 'test':
		testing_data = [line.strip('\n').split(',') for line in open(args.testing_path).readlines()]
		sample_id = np.asarray(testing_data)[1:, 0].astype(str)
		x_pre = (np.array(testing_data)[1:, 1:]).astype(float)
		#print(x_pre)

		x_user = x_pre[:, 0].reshape(-1,1)
		#print("x_user: ", x_user)
		
		x_movie = x_pre[:, 1].reshape(-1,1)
		#print("x_movie: ", x_movie)

		one_matrix = np.ones(len(x_movie))

		#print(x_user.shape)
		#print(x_movie.shape)
		#print(one_matrix.shape)
		y_pre = testing(x_user, x_movie, one_matrix, args.model_load_path)
		print("done")
		y_pre = y_pre 
		df_output = pd.DataFrame(y_pre, columns = ["Rating"], index = sample_id)
		#print(df_output)
		#df_output.index.name = "id"
		df_output.to_csv(args.prediction_save_path, index_label = 'TestDataID')
		#print(np.around(y_pre).aastype(str))

	if args.action == 'train':
		train_model()




if __name__ == '__main__':
	main()
