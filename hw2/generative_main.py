import data_preprocessing as DP
import numpy as np
import pandas as pd
import math

##C1 is type 1 here
PATH_X_TRAIN_C1 = "X_train_type_1.csv"
PATH_X_TRAIN_C2 = "X_train_type_0.csv"
PATH_X_TRAIN_ALL = "X_train.csv"
PATH_X_TEST = "X_test.csv"

ALL_NUM_DATA_C1 = 7841
ALL_NUM_DATA_C2 = 24720
ALL_NUM_FEATURE = 106
ALL_NUM_DATA_TEST = 16281


NUM_ROW_WANTED_C1 = int(ALL_NUM_DATA_C1)
NUM_ROW_WANTED_C2 = int(ALL_NUM_DATA_C2)

ROW_WANTED_C1 = range(NUM_ROW_WANTED_C1)
ROW_WANTED_C2 = range(NUM_ROW_WANTED_C2)

ROW_WANTED_PRE_C1 = range(ALL_NUM_DATA_C1 - NUM_ROW_WANTED_C1, ALL_NUM_DATA_C1)
ROW_WANTED_PRE_C2 = range(ALL_NUM_DATA_C2 - NUM_ROW_WANTED_C2, ALL_NUM_DATA_C2)

COL_WANTED = range(ALL_NUM_FEATURE)

x_train_C1 = DP.get_data(PATH_X_TRAIN_C1, 0, ALL_NUM_FEATURE, 0, NUM_ROW_WANTED_C1)
x_train_C2 = DP.get_data(PATH_X_TRAIN_C2, 0, ALL_NUM_FEATURE, 0, NUM_ROW_WANTED_C2)
x_all = (pd.read_csv(PATH_X_TRAIN_ALL)).values



x_test = DP.get_data(PATH_X_TEST, 0, ALL_NUM_FEATURE, 0, ALL_NUM_DATA_TEST)
#print(x_test)

N1 = NUM_ROW_WANTED_C1
N2 = NUM_ROW_WANTED_C2

mean_all = np.mean(x_test, axis = 0)
var_all = np.var(x_test, axis = 0)


x_train_C1_normalized = DP.normalization(x_train_C1, mean_all, var_all)
x_train_C2_normalized = DP.normalization(x_train_C2, mean_all, var_all)

x_test_normalized = DP.normalization(x_test, mean_all, var_all)


mean_C1_normalized = np.mean(x_train_C1_normalized, axis=0)
mean_C2_normalized = np.mean(x_train_C2_normalized, axis=0) 


sigma_C1 = np.cov(x_train_C1_normalized.T)
sigma_C2 = np.cov(x_train_C2_normalized.T)


sigma = (N1*sigma_C1 + N2*sigma_C2)/(N1+N2)

w = DP.get_w(mean_C1_normalized, mean_C2_normalized, sigma)
b = DP.get_b(mean_C1_normalized, mean_C2_normalized, sigma, N1, N2)

w.shape = 1, 106

b = np.array([b])
df_w = pd.DataFrame(w)
df_b = pd.DataFrame(b)

df_w.to_csv("W_generaitve.csv", header=False, index=False)
df_b.to_csv("b_generative.csv", header=False, index=False)


'''
z = DP.get_z(w, b, x_test_normalized)



y_pre_C1 = np.array([])

for i in range(len(z)):

	if DP.sigmoid(z[i]) >= 0.5:
		y_pre_C1 = np.append(y_pre_C1, "1")
	
	else: 
		y_pre_C1 = np.append(y_pre_C1, "0")
	

	#print(pre)
	#y_pre_C1 = np.append(y_pre_C1, DP.predict(x_test_normalized[i], mean_C1_normalized, mean_C2_normalized, sigma, N1, N2))


idx = []
for i in range(len(y_pre_C1)):
	idx.append(i+1)
df_y_pre = pd.DataFrame(y_pre_C1, index = idx)


print("done")
df_y_pre.to_csv("Y_PRE.csv", index = True, header = ["label"], index_label = "id")
'''