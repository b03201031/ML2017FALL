# -*- coding: utf-8 -*-

import csv
import codecs
import numpy as np
import pandas as pd

NUM_OBJ = 18
DATA_BEG = 3
DATA_END = 26
NUM_DATA_ONE_SET = 24
NUM_TOTAL_LINE = 4320
NUM_DATASET = 4320 / NUM_OBJ

NUM_ROW = NUM_DATA_ONE_SET
NUM_COL = NUM_TOTAL_LINE

full_table = pd.read_csv('train.csv',encoding = 'ISO-8859-1')

full_table.drop(full_table.columns[:3], axis=1, inplace=True)



tmp = [full_table[i*18:(i+1)*18].reset_index() for i in range(int(len(full_table.index)/18))]

all_data = pd.concat(tmp, axis=1 ).values


print(all_data[0,-24:])
'''
with codecs.open("train.csv", "r",encoding='utf-8', errors='ignore') as f:
	reader = csv.reader(f)
	my_list = list(reader)
	del my_list[0]

new_list = [ls[3:] for ls in my_list]

data = np.concatenate([np.array(i) for i in new_list])

print(len(data))

data.shape = NUM_ROW, NUM_COL
data_y = data
'''
