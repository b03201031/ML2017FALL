import numpy as np
import pandas as pd

full_table = pd.read_csv("train.csv", encoding = 'ISO-8859-1')
full_table.drop(full_table.columns[0:3], axis = 1, inplace = True)
full_table.replace({'NR':0}, inplace = True)
full_table = full_table.apply(pd.to_numeric)



tmp_table = [full_table[18*i:18*(i+1)].reset_index(drop=True) for i in range(int(len(full_table.index)/18))]

print(tmp_table[0])