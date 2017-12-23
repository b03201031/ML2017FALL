#!/bin/bash
wget https://www.dropbox.com/s/f1fgj6tdf1rpf90/keras_model_epoch_08_vloss_0.75.hdf5?dl=1 -O model_hw5.hdf5

python3 hw5.py \
--action "test" \
--training_path "./train.csv" \
--testing_path $1 \
--model_load_path "./model_hw5.hdf5" \
--prediction_save_path $2 \
