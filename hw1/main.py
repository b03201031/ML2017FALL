import csv
import codecs
import sys
import numpy as np
import pandas as pd
import derivative as fc
import matplotlib.pyplot as plt
import matplotlib as mpl

PIC_TITLE = "18features"
f_path_in = sys.argv[1]
f_path_out = sys.argv[2]

NUM_DATA_SET = 470
NUM_MONTH_TAKED = 12
NUM_DURATION = 9
LEARNING_RATE = 0.3
#[0, 2, 5, 7, 8, 9, 12]
data_selected = np.arange(18)
IDX_PM25 = 9
DIM = 1

full_table = pd.read_csv(f_path_in, encoding = 'ISO-8859-1')
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
#x_train = np.concatenate((x_train), axis=1)
#x_test = np.concatenate((x_test, x_test*x_test), axis=1)


output_set = np.array([[]])
## FS
M_x = fc.Mean(x_train)
SD_x = fc.SD(x_train)
output_set = np.append(output_set, 0)
output_set = np.append(output_set, M_x)
output_set = np.append(output_set, 1)
output_set = np.append(output_set, SD_x)
print(len(output_set))
output_set.shape = 2, len(x_test[0, :])+1

for i in range(len(x_train[0, :])):
	x_train[:, i] = (x_train[:, i] - M_x[i]) / SD_x[i]






for i in range(len(x_test[0, :])):
	x_test[:, i] = (x_test[:, i] - M_x[i]) / SD_x[i]



x_train = np.concatenate((np.reshape(np.repeat(1, len(x_train)), (len(x_train), 1)),x_train), axis=1)
#x_test = np.concatenate((np.reshape(np.repeat(1, len(x_test)), (len(x_test), 1)),x_test), axis=1)



 # set theta_0 to [1,1,1,1.....]
theta_0_1 = np.repeat(0.5, len(x_train[0,:]))
theta_0_2 = theta_0_1
theta_0_3 = theta_0_1
theta_0_4 = theta_0_1


rmse_set = []


ada_1 = 0
ada_2 = 0
ada_3 = 0
ada_4 = 0
lamda = 0.1

loss_set_1 = []
loss_set_2 = []
loss_set_3 = []
loss_set_4 = []

NUM_ITER = NUM_DATA_SET*NUM_MONTH_TAKED
for i in range(NUM_ITER):
	theta_1_1, ada_1 = fc.Gradient_Decent(theta_0_1, LEARNING_RATE , x_train, y_train, ada_1, i+1, 0.1)
	theta_1_2, ada_2 = fc.Gradient_Decent(theta_0_2, LEARNING_RATE , x_train, y_train, ada_2, i+1, 0.01)
	theta_1_3, ada_3 = fc.Gradient_Decent(theta_0_3, LEARNING_RATE , x_train, y_train, ada_3, i+1, 0.001)
	theta_1_4, ada_4 = fc.Gradient_Decent(theta_0_4, LEARNING_RATE , x_train, y_train, ada_4, i+1, 0.0001)
	#rmse_set.append(fc.RMSE(x_test, y_test, theta_1))
	loss_set_1.append(fc.Loss(theta_1_1, x_train, y_train, 0.1)/NUM_ITER)	
	loss_set_2.append(fc.Loss(theta_1_2, x_train, y_train, 0.01)/NUM_ITER)
	loss_set_3.append(fc.Loss(theta_1_3, x_train, y_train, 0.001)/NUM_ITER)
	loss_set_4.append(fc.Loss(theta_1_4, x_train, y_train, 0.0001)/NUM_ITER)



	if i == NUM_ITER-1:
		output_set = np.append(output_set, theta_1_1, axis = 0)

	theta_0_1 = theta_1_1
	theta_0_2 = theta_1_2
	theta_0_3 = theta_1_3
	theta_0_4 = theta_1_4




df_out = pd.DataFrame(output_set)
#print(df_out)
df_out.to_csv(f_path_out, index = False)

#df_rmse = pd.DataFrame(rmse_set)
#df_rmse.to_csv("rmse.csv", index = False)

prediction_value = []


#print(fc.RMSE(x_test, y_test, beta_set[len(beta_set)-1, :]))
x_plot = list(range(NUM_ITER))

mpl.style.use("default")
fig, ax = plt.subplots(figsize=(3,3))
ax.set_title("duration=9, all ")
ax.plot(loss_set_1, 'r', label="0.1")
ax.plot(loss_set_2, 'y', label="0.01")
ax.plot(loss_set_3, 'b', label="0.001")
ax.plot(loss_set_4, 'g', label="0.0001")
ax.legend()
plt.show()


#print(full_table.index)
