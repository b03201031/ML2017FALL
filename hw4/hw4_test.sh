#!/bin/bash
wget https://www.dropbox.com/s/nvda0kehoozqsuo/cnn_model_epoch_39_vloss_0.43.hdf5?dl=1 -O model_3.hdf5 &&
wget https://www.dropbox.com/s/693e41jhdthsrng/gensim_model_125?dl=1 -O gensim_model_125 &&
	
python3 ensemble_predict.py $1 $2
