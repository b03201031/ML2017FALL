#!/bin/bash

python3 chat-bot.py \
--action 'test' \
--training_data_path './training_data/4_train.txt' \
--testing_data_path 'testing_data.csv' \
--jieba_dic_path './training_data/dict.txt.big.txt' \
--w2v_model_save_path './word2vec_model/w2v_cutall_' \
--w2v_model_load_path './word2vec_model/w2v_cutall_' \
--w2v_load_option "" \
--w2v_train_cutall "" \
--w2v_vec_dim 64 \
--w2v_min_count 0 \

