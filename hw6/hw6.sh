#!/bin/bash
wget https://www.dropbox.com/s/bqhqxo414hrzzzr/nn_encoder.h5?dl=1 -O nn_encoder.h5 
wget https://www.dropbox.com/s/zpwir2tv3uojb6q/nn_autoencoder.h5?dl=1 -O nn_autoencoder.h5
python3 hw6.py $1 $2 $3