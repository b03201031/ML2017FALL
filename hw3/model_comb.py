##test normalizeation
import os
import numpy as np 
import pickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.merge import add, concatenate
from keras.layers import normalization, Average, Add
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, Input, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Merge, Concatenate, Multiply, Subtract
from keras.optimizers import SGD, Adam, rmsprop, Adadelta
from keras.callbacks import EarlyStopping
import sys


BATCH_SIZE = 96
EPOCHS = 40
input_shape = (48, 48, 1)



FILE_PATH = sys.argv[1]

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


    
    block1_1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1_1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1_1)
    block1_1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1_1)
    block1_1 = BatchNormalization()(block1_1)
    block1_1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1_1)

    block1_2 = Conv2D(64, (3, 3), activation='relu')(block1_1)
    block1_2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1_2)

    block1_3 = Conv2D(64, (3, 3), activation='relu')(block1_2)
    block1_3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block1_3)
    block1_3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1_3)

    block1_4 = Conv2D(128, (3, 3), activation='relu')(block1_3)
    block1_4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1_4)

    block1_5 = Conv2D(128, (3, 3), activation='relu')(block1_4)
    block1_5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1_5)
    block1_5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block1_5)
    block1_5 = Flatten()(block1_5)

    fc1_1 = Dense(1024, activation='relu')(block1_5)
    fc1_1 = Dropout(0.5)(fc1_1)

    fc1_2 = Dense(512, activation='relu')(fc1_1)
    fc1_2 = Dropout(0.5)(fc1_2)

    predict1 = Dense(7)(fc1_2)
    predict1 = Activation('softmax')(predict1)



    input_img_2 = Input(shape=(48, 48, 1))
    block2_1 = Conv2D(96, (5, 5), activation = 'relu')(input_img) 
    block2_1 = AveragePooling2D((2,2))(block2_1)

    block2_2 = Conv2D(96,(5, 5),   activation = 'relu')(block2_1)
    block2_2 = MaxPooling2D((2,2))(block2_2)
    block2_2 = BatchNormalization()(block2_2)
    #block2_2 = Dropout(0.5)(block2_2)

    block2_3 = Conv2D(96, (3, 3), activation='relu')(block2_2)
    block2_3 = MaxPooling2D((2,2))(block2_3)


    flat2 = Flatten()(block2_3)

    fc2_1 = Dense(1024, activation='relu')(flat2)
    fc2_1 = Dropout(0.5)(fc2_1)

    fc2_2 = Dense(512, activation='relu')(fc2_1)
    fc2_2 = Dropout(0.5)(fc2_2)

    predict2 = Dense(7)(fc2_2)
    predict2 = Activation('softmax')(predict2)

    aver = Average()([predict1, predict1, predict1, predict2, predict2])
    #merge_pre = Concatenate()([predict1, predict2])

    #reg = Dense(units = 7, activation='linear')(merge_pre)



    model = Model(inputs=input_img, outputs=aver)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model




def build_generator(x_train):
    
    datagen = ImageDataGenerator(
        #zca_whitening=True,
        rotation_range=10,
        width_shift_range=0.3, 
        height_shift_range=0.3,
        fill_mode = "nearest",
        horizontal_flip = True
        )
    datagen.fit(x_train)

    return datagen



file_path = FILE_PATH
#data_path = os.environ.get("GRAPE_DATASET_DIR")
#file_path = data_path+'/data/train.csv'


x_raw, label = load_data(file_path)

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

#model = load_model("./sample_comb2_120.h5")
model = build_model()


#earlyStopping=EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='auto')

#history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs=EPOCHS, validation_data=[x_val, y_val])
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size = BATCH_SIZE),
    steps_per_epoch=int(len(x_train)*2.25 / BATCH_SIZE),
    epochs = EPOCHS,
    validation_data = (x_val, y_val))

model.save("sample_comb2_160.h5")