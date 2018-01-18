#!/bin/bash
wget https://www.dropbox.com/s/ut8gqa3uisrt0hr/cutted_ques_search.pkl?dl=1 -O cutted_ques.pkl
wget https://www.dropbox.com/s/nh3hkqzzbd6xnzl/cutted_ans_search.pkl?dl=1 -O cutted_ans.pkl
wget https://www.dropbox.com/s/wxr1flyfqvusoij/test_id?dl=1 -O test_id
wget https://www.dropbox.com/s/z9v9oxmre9ur96h/w2v_nocutall_256?dl=1 -O model.h5
wget https://www.dropbox.com/s/abpdkjhbhv7phxz/w2v_nocutall_256.wv.syn0.npy?dl=1 -O model.h5.wv.syn0.npy
wget https://www.dropbox.com/s/l0uuqmk1m717ujc/w2v_nocutall_256.syn1neg.npy?dl=1 -O model.h5.syn1neg.npy
python3 predict.py $1