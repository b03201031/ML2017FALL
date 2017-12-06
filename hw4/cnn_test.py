import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.layers import normalization, Average, Add, Bidirectional, GRU, SimpleRNN, TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization, LSTM
from keras.layers import Dense, Merge, Concatenate, Multiply, Subtract, Masking
from keras.optimizers import SGD, Adam, rmsprop, Adadelta
from keras.callbacks import ModelCheckpoint

# Steady setting : 
'''
conv_lens = [2,3,5]
conv_dim = 128
dnn_dim = [32,64]
'''

conv_lens = [2,3,5]
conv_dim = 128
dnn_dim = [32,64]


tr_x = np.load("./preprocessed_data/texts_train_125.npy")
tr_x = tr_x.reshape((-1,40,125,1))
tr_y = np.load("./preprocessed_data/label_train.npy")
new_tr_y = np.zeros((tr_y.shape[0],2))
for idx,i in enumerate(tr_y):
	new_tr_y[idx,i] = 1 
#print(tr_y.shape)

saver = ModelCheckpoint('./models/cnn_model_epoch_{epoch:02d}_vloss_{val_loss:.2f}.hdf5', monitor='val_loss',save_best_only=True)

w2v_input = Input(shape=(40,125,1))

cnn_features = []
for filter_len in conv_lens:
	conv_out = Conv2D(conv_dim,kernel_size=(filter_len,125),activation='relu')(w2v_input)
	pooled = MaxPooling2D((40 - filter_len+1,1))(conv_out)

	cnn_features.append(pooled)
conv_features = concatenate(cnn_features)
flatten_features = Flatten()(conv_features)

dnn_features = [flatten_features]
for dim in dnn_dim:
	drop = Dropout(0.5)(dnn_features[-1])
	hidden_layer = Dense(dim,activation='relu')(drop)
	dnn_features.append(hidden_layer)

drop = Dropout(0.5)(dnn_features[-1])
output_layer = Dense(2,activation='softmax')(drop)

model = Model(inputs=w2v_input, outputs=output_layer)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(tr_x,new_tr_y,epochs=50,batch_size=256,validation_split=.1,callbacks=[saver])
