import csv
import codecs
import numpy as np
import pandas as pd
import derivative as fc
import matplotlib.pyplot as plt


f_path = sys.argv[1]

NUM_DATA_SET = 470
NUM_MONTH_TAKED = 11
NUM_DURATION = 5
LEARNING_RATE = 0.6
#[0, 2, 5, 7, 8, 9, 12]
data_selected = np.array([0, 2, 5, 7, 8, 9, 12])
NUM_VAR = 2
IDX_PM25 = 9
DIM = 1

full_table = pd.read_csv(f_path, encoding = 'ISO-8859-1')
full_table.drop(full_table.columns[0:3], axis = 1, inplace = True)
full_table.replace({'NR':0}, inplace = True)
full_table = full_table.apply(pd.to_numeric)



tmp_table = [full_table[18*i:18*(i+1)].reset_index(drop=True) for i in range(int(len(full_table.index)/18))]
all_data = pd.concat(tmp_table, axis=1).values


x_test, y_test, x_train, y_train = np.array([[]]), np.array([[]]), np.array([[]]), np.array([[]])


for i in range(1, NUM_MONTH_TAKED+1):
	
	tmp_x_train, tmp_y_train = fc.DATA_PREPROCESSING(all_data, NUM_DATA_SET, NUM_DURATION, IDX_PM25, i, data_selected)

	
	if np.size(x_train) == 0:
		x_train = tmp_x_train
		y_train = tmp_y_train

		
	else:
		x_train = np.concatenate((x_train, tmp_x_train), axis=0)
		y_train = np.append(y_train, tmp_y_train)
		
	

x_test, y_test = fc.DATA_PREPROCESSING(all_data, NUM_DATA_SET, NUM_DURATION, IDX_PM25, 12, data_selected)	


# x x^2 x^3
x_train_8 = x_train[:, (NUM_DURATION)*5:NUM_DURATION*6]
x_train_9 = x_train[:, (NUM_DURATION)*6:NUM_DURATION*7]
x_test_8 = x_test[:, (NUM_DURATION)*5:NUM_DURATION*6]
x_test_9 = x_test[:, (NUM_DURATION)*6:NUM_DURATION*7]
print(x_test_9.shape)
x_train = np.concatenate((x_train, x_train_8*x_train_8, x_train_9*x_train_9), axis=1)
x_test = np.concatenate((x_test, x_test_8*x_test_8, x_test_9*x_test_9), axis=1)



## FS
M_x = fc.Mean(x_train)
SD_x = fc.SD(x_train)

for i in range(len(x_train[0, :])):
	x_train[:, i] = (x_train[:, i] - M_x[i]) / SD_x[i]




M_SD = pd.DataFrame(np.array([M_x, SD_x]))
M_SD.to_csv("M_SD.csv", header = None, index=False)


for i in range(len(x_test[0, :])):
	x_test[:, i] = (x_test[:, i] - M_x[i]) / SD_x[i]



x_train = np.concatenate((np.reshape(np.repeat(1, len(x_train)), (len(x_train), 1)),x_train), axis=1)
x_test = np.concatenate((np.reshape(np.repeat(1, len(x_test)), (len(x_test), 1)),x_test), axis=1)



 # set theta_0 to [1,1,1,1.....]
theta_0 = np.repeat(0, len(x_train[0,:]))



rmse_set = []
beta_set = np.array([[]])
loss_set = []
ada = 0
for i in range(NUM_DATA_SET*NUM_MONTH_TAKED):
	theta_1, ada = fc.Gradient_Decent(theta_0, LEARNING_RATE , x_train, y_train, ada, i+1)
	rmse_set.append(fc.RMSE(x_test, y_test, theta_1))
	loss_set.append(fc.Loss(theta_1, x_train, y_train)/(NUM_MONTH_TAKED*NUM_DATA_SET))
	#print(fc.Loss(theta_1, x, y))
	beta_set = np.append(beta_set, theta_1)
	theta_0 = theta_1

beta_set.shape = NUM_DATA_SET*NUM_MONTH_TAKED, len(data_selected)*NUM_DURATION*DIM+2*NUM_DURATION+1

df_beta = pd.DataFrame(beta_set)
df_beta.to_csv("beta.csv", index = False)

df_rmse = pd.DataFrame(rmse_set)
df_rmse.to_csv("rmse.csv", index = False)

prediction_value = []

for x_0 in x_test:
	prediction_value.append(fc.Regression(beta_set[len(beta_set)-1], x_0))

df_pre = pd.DataFrame(prediction_value)

df_pre.to_csv("main_pre.csv")
print(fc.RMSE(x_test, y_test, beta_set[len(beta_set)-1, :]))
print(loss_set[len(loss_set)-1])
plt.plot(loss_set)
plt.show()


#print(full_table.index)
