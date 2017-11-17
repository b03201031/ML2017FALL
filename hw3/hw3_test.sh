#!/bin/bash
wget https://www.dropbox.com/s/ox18g4b2xhr90dn/model_sample.h5?dl=1 -O model_sample.h5 &&
wget https://www.dropbox.com/s/snvaf191awcw82a/model_comb.h5?dl=1 -O model_comb.h5 &&
python3 model_voting.py $1 $2