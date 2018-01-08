from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys

def get_data():
	train_num = 130000
	X = np.load('image.npy')
	X = X.astype('float32') / 255.
	X = np.reshape(X, (len(X), -1))
	x_train = X[:train_num]
	x_val = X[train_num:]

	return x_train, x_val

def build_model():
	
	'''
	input_img = Input(shape=(28, 28, 1))

	x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same')(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same')(x)

	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, 3 ,3, activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

	autoencoder = Model(input_img, decoded)
	encoder = Model(input_img, encoded)

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	'''
	
	input_img = Input(shape=(784, ))
	#encoded = Dense(512, activation='relu')(input_img)
	encoded = Dense(256, activation='relu')(input_img)
	#encoded = Dense(98, activation='relu')(encoded)

	encoded = Dense(64)(encoded)

	decoded = Dense(256, activation='relu')(encoded)
	#decoded = Dense(512, activation='relu')(decoded)
	#decoded = Dense(784, activation='relu')(decoded)
	decoded = Dense(784, activation='sigmoid')(decoded)

	encoder = Model(input=input_img, output=encoded)

	autoencoder = Model(input=input_img, output=decoded)
	adam = Adam(lr=0.0005)
	autoencoder.compile(optimizer=adam, loss='mse')
	
	return encoder, autoencoder

def plot(X, label):
	from matplotlib import pyplot as  plt
	from sklearn.manifold import TSNE
	label = np.array(label)
	X = np.array(X, dtype=np.float64)
	print(X.shape)
	vis_data = TSNE(n_components=2, verbose=1).fit_transform(X)
	np.save('vis_data', vis_data)
	vis_x = vis_data[:, 0]
	vis_y = vis_data[:, 1]

	print("ploting")
	cm = plt.cm.get_cmap('RdBu')
	sc = plt.scatter(vis_x, vis_y, c=label, cmap=cm)
	#plt.colorbar(sc)
	plt.show()

def main():
	train_num = 130000
	X = np.load(sys.argv[1])
	X = X.astype('float32') 
	X = np.reshape(X, (len(X), -1))
	X_maen = np.mean(X, axis=0)
	print(X_maen.shape)
	X_maen = np.reshape(X_maen, (1, len(X_maen)))
	X = X - X_maen
	X = X / 255.
#	X = np.reshape(X, (len(X), 28, 28 , 1))
	
	#print(X[0].shape)
	x_train = X[:train_num]
	x_val = X[train_num:]
	
	autoencoder = load_model('nn_autoencoder.h5')
	encoder = load_model('nn_encoder.h5')
	autoencoder.summary()
	#autoencoder.summary()
	encoded_imgs = encoder.predict(X)
	encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
	print(encoded_imgs.shape)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)
	'''
	label = []
	
	for idx in range(500):
		label.append(0)
	for idx in range(500):
		label.append(1)
	
	print(label[:100])
	print(np.concatenate((encoded_imgs[:5000], encoded_imgs[-5000:])).shape)
	plot(np.concatenate((encoded_imgs[:500], encoded_imgs[-500:])), label)
	'''
	f = pd.read_csv(sys.argv[2])
	IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
	o = open(sys.argv[3], 'w')
	o.write('ID,Ans\n')
	for idx, i1, i2 in zip(IDs, idx1, idx2):
		p1 = kmeans.labels_[i1]
		p2 = kmeans.labels_[i2]
		if p1 == p2:
			pred = 1
		else:
			pred = 0
		o.write('{},{}\n'.format(idx, pred))
	o.close()
	
if __name__ == '__main__':
	main()