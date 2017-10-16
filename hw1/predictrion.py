import numpy as np
import pandas as pd
import derivative as fc
import sys


f_path_in = sys.argv[1]
f_path_out = sys.argv[2]

NUM_DURATION = 9

data_selected = np.arange(18)
NUM_ELEMENT_SELECTED = len(data_selected)
df_para = pd.read_csv("out_18.csv")


#full_beta = full_beta.apply(pd.to_numeric)
beta = df_para.values[2,:]


full_data = pd.read_csv(f_path_in, header = None)
full_data.drop(full_data.columns[0:2], axis=1, inplace=True)
full_data.replace({'NR':0}, inplace=True)
full_data = full_data.apply(pd.to_numeric)

tmp_ls_data = [full_data[i*18: (i+1)*18].reset_index(drop=True) for i in range(int(len(full_data)/18))]

x = np.array([[]])



#x.shape = int(len(full_data)/18), NUM_ELEMENT_SELECTED*NUM_DURATION

for df in tmp_ls_data:
	for i in data_selected:
		x = np.append(x, df.values[i, 9-NUM_DURATION:])
	

x.shape = int(len(full_data)/18), NUM_DURATION*NUM_ELEMENT_SELECTED


#x = np.concatenate((x), axis=1)
#x = np.concatenate((x,x*x*x), axis=1)

M = df_para.values[0, 1:]

SD = df_para.values[1, 1:]

print(len(M))
for i in range(len(x[0,:])):
	x[:, i] = (x[:, i] - M[i])/SD[i]
	


x = np.concatenate((np.reshape(np.repeat(1, len(x)), (len(x), 1)),x), axis=1)
#print(x)


y = np.array([[]])


col_beta = np.reshape(beta, (len(beta), 1))
for x_0 in x:
	#print(np.dot(x_0, col_beta))

	y = np.append(y, np.dot(x_0, col_beta))




df_y = pd.DataFrame(y)
df_y.columns = ["value"]



ls_id = []
for i in range(len(df_y.index)):
	ls_id.append("id_"+str(i)) 

df_y.index = ls_id
df_y.rename_axis = "id"


df_y.to_csv(f_path_out, index_label = "id")
