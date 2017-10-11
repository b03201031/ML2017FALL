import numpy as np
import pandas as pd
import derivative as fc

NUM_DURATION = 5
NUM_ELEMENT_SELECTED = 7
data_selected = np.array([0, 2, 5, 7, 8, 9, 12])
full_beta = pd.read_csv("beta.csv", header = None)
full_beta = full_beta.apply(pd.to_numeric)
beta = full_beta.values[len(full_beta.values)-1, :]

beta.shape = np.size(beta)

full_data = pd.read_csv("test.csv", header = None)
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
print(len(x[:,0]))

x_8 = x[:, NUM_DURATION*5:NUM_DURATION*6]
x_9 = x[:, NUM_DURATION*6:NUM_DURATION*7]

x = np.concatenate((x,x_8**2,x_9**2), axis=1)
#x = np.concatenate((x,x*x*x), axis=1)
print(len(x[:,0]))
print(x)

M_SD = pd.read_csv("M_SD.csv", header=None)
M_SD = M_SD.apply(pd.to_numeric)
M_SD_np = M_SD.values
print(M_SD_np.shape)

M = M_SD_np[0]
SD = M_SD_np[1]

print(len(x[:,0]))


for i in range(len(x[0,:])):
	x[:, i] = (x[:, i] - M[i])/SD[i]
	


x = np.concatenate((np.reshape(np.repeat(1, len(x)), (len(x), 1)),x), axis=1)
#print(x)


y = np.array([[]])


col_beta = np.reshape(beta, (len(beta), 1))
for x_0 in x:
	print(np.dot(x_0, col_beta))

	y = np.append(y, np.dot(x_0, col_beta))




df_y = pd.DataFrame(y)
df_y.columns = ["value"]



ls_id = []
for i in range(len(df_y.index)):
	ls_id.append("id_"+str(i)) 

df_y.index = ls_id
df_y.rename_axis = "id"


df_y.to_csv("test_output_04.csv")
