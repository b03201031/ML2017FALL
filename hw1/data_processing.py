import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df_data = pd.read_csv("train.csv", encoding = 'ISO-8859-1')
df_data.drop(df_data.columns[0:3], axis = 1, inplace = True)
df_data.replace({'NR':0}, inplace=True)
df_data = df_data.apply(pd.to_numeric)
tmp_ls = [df_data[i*18: (i+1)*18].reset_index(drop=True) for i in range(int(len(df_data)/18))]
np_data = pd.concat(tmp_ls, axis=1).values

data_month = [np_data[:, i*480:(i+1)*480] for i in range(int(len(np_data[0, :])/480))]

trend_pm_month = []
for data in data_month:
	
	tmp_arr = np.array([])
	for i in range(480-1):
		#decreasing
		if data[9, i] > data[9, i+1]:
			tmp_arr = np.append(tmp_arr, -1)
		#equal
		elif data[9, i] == data[9, i+1]:
			tmp_arr = np.append(tmp_arr, 0)
		#increasing
		elif data[9, i] < data[9, i+1]:
			tmp_arr = np.append(tmp_arr, 1)

	tmp_arr = np.append(tmp_arr, 0)

	trend_pm_month.append(tmp_arr)

for data in data_month:
	data = np.reshape(data, (18, 480))
p1 = plt.figure()
plt.plot(data_month[0][8, 0:479], data_month[0][9, 1:480], 'ro')
p1.savefig("8.png")

'''
for ele in trend_pm_month[9]:
	if ele == 0:
		pass
	elif ele == -1:
		tmp_value = tmp_value -1
	elif ele == 1:
		tmp_value = tmp_value + 1


	plt.plot(idx, tmp_value, 'ro')

	idx = idx + 1
	if idx%24 == 0:
		tmp_value = 0

plt.show()
'''