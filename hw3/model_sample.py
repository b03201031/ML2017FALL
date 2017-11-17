##test normalization
#acc_65

import numpy as np 
import pandas as pd

import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.layers import normalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Merge, Concatenate
from keras.optimizers import SGD, Adam, rmsprop, Adadelta
from keras.callbacks import EarlyStopping
import sys

FILE_PATH = sys.argv[1]



BATCH_SIZE = 96
EPOCHS = 40 ## best
input_shape = (48, 48, 1)



def load_data(file_path):

    with open(file_path, 'r') as f:
        f.readline()
        str_pixel = ""
        str_label = ""
    
        for i, line in enumerate(f):
            data = line.split(',')
            label = data[0]
            pixel = data[1]
    
            str_pixel = str_pixel + pixel
            str_pixel = str_pixel + ' '
            str_label = str_label + label
            str_label = str_label + ' '

        
        arr_pixel = np.array(str_pixel[:-2].split(' '))
        arr_label = np.array(str_label[:-1].split(' '))
        arr_pixel = arr_pixel.astype(int)
        arr_label = arr_label.astype(int)
        arr_pixel = arr_pixel.reshape(len(arr_label), 48*48)

        return arr_pixel, arr_label 


def build_model():

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = BatchNormalization()(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.3)(fc1)

    fc2 = Dense(512, activation='relu')(fc1)
    fc2 = Dropout(0.3)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model




def build_generator(x_train):
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.3, 
        height_shift_range=0.3,
        fill_mode = "nearest",
        horizontal_flip = True
        )
    datagen.fit(x_train)

    return datagen






x_raw, label = load_data(FILE_PATH)

y_raw = np.zeros((len(label),7))


for i in range(len(label)):
    y_raw[i, label[i]] = 1



##normalization
x_raw = x_raw /255

x_raw = x_raw.reshape(x_raw.shape[0], 48, 48, 1)

x_train = x_raw[:int(len(x_raw)*0.8)]
y_train = y_raw[:int(len(y_raw)*0.8)]


x_val = x_raw[int(len(x_raw)*0.8) :]
y_val = y_raw[int(len(y_raw)*0.8) :]
datagen = build_generator(x_train)


model = build_model()
'''
x_test ,sample_id = load_data(FILE_PATH)


x_test = x_test/255
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
model = load_model("./sample_augment_5_120.h5")
y_pre = model.predict(x_test)

y_test = np.array([])

for i in y_pre:
    y_test = np.append(y_test, np.argmax(i).astype(str))
 
  
df_output = pd.DataFrame(y_test, columns = ["label"], index = sample_id)
df_output.index.name = "id"

df_output.to_csv("first_pre.csv", index_label = 'id')
'''

#earlyStopping=EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='auto')

#history = model.fit(x_raw, y_raw, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
model.fit_generator(datagen.flow(x_train, y_train, batch_size = BATCH_SIZE), steps_per_epoch=int(len(x_train)*2.25 / BATCH_SIZE), epochs = EPOCHS, validation_data = (x_val, y_val))

#with open('./history_aug5_130', 'wb') as file_pi:
 #   pickle.dump(history.history, file_pi)

model.save("model_sample.h5")
