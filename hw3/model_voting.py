##test normalizeation
import os
import pandas as pd
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



FILE_PATH_INPUT = sys.argv[1]
FILE_PATH_OUTPUT = sys.argv[2]

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
        arr_pixel = arr_pixel/255
        arr_pixel = arr_pixel.reshape(arr_pixel.shape[0], 48, 48, 1)
        

        return arr_pixel, arr_label 


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



def voting(models, testing_data):
    model_1 = models[0]
    y_pre = model_1.predict(testing_data)

    for i in range(1, len(models)):
        y_pre = y_pre + models[i].predict(testing_data)

    y_test = np.array([])

    for i in y_pre:
        y_test = np.append(y_test, np.argmax(i).astype(str))

    return y_test

def evaluate(y_pre, y_val):
    count = 0.0
    for i in range(len(y_pre)):
        if int(y_pre[i]) == int(y_val[i]):
            count = count + 1

    print(count)
    return count/float(len(y_val))

def main():
    #x_test ,sample_id = load_data("test.csv")


    
    #x_raw, y_raw = load_data("train.csv")
    #x_raw = x_raw/255
    #x_raw = x_raw.reshape(x_raw.shape[0], 48, 48, 1)
    #print(len(x_raw))
    #print(len(y_raw))
    #print(x_raw.shape)
    #x_val = x_raw[int(0.9*len(x_raw)):]
    #y_val = y_raw[int(0.9*len(y_raw)):]


    x_test, sample_id = load_data(FILE_PATH_INPUT)
    path_model = ["model_sample.h5","model_comb.h5"]
    models = np.array([])

    for path in path_model:
        models = np.append(models, load_model(path))

    y_test = voting(models, x_test)
    #print(x_val.shape)
    #print(y_test)
    #print(y_val)

    #print(evaluate(y_test, y_val))

    
    df_output = pd.DataFrame(y_test, columns = ["label"], index = sample_id)
    df_output.index.name = "id"

    df_output.to_csv(FILE_PATH_OUTPUT, index_label = 'id')
    

main()