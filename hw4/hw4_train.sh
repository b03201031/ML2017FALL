#!/bin/bash

wget https://www.dropbox.com/s/693e41jhdthsrng/gensim_model_125?dl=1 -O gensim_model_125 &&
	
python3 training.py $1 $2