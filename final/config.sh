#!/bin/bash

python3 chat-bot.py \
--action 'train' \
--testing_data_path 'testing_data.csv' \
--jieba_dic_path './training_data/dict.txt.big.txt' \
--w2v_load_option "" \
--w2v_train_cutall "" \
--search "" \
--w2v_vec_dim 64 \
--w2v_min_count 1 \
--iter 20 \
--sg 1 \