import data_preprocessing as DP
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

##C1 is type 1 here
PATH_X_TRAIN_C1 = "X_train_type_1.csv"
PATH_X_TRAIN_C2 = "X_train_type_0.csv"
PATH_X_TRAIN_ALL = "X_train.csv"
PATH_X_TEST = "X_test.csv"


########################################constant declare
ALL_NUM_DATA_C1 = 7841
ALL_NUM_DATA_C2 = 24720
ALL_NUM_FEATURE = 106
ALL_NUM_DATA_TEST = 16281
LEARNING_RATE = 0.001
NUM_ITE = ALL_NUM_DATA_C1 + ALL_NUM_DATA_C2


NUM_ROW_WANTED_C1 = int(ALL_NUM_DATA_C1)
NUM_ROW_WANTED_C2 = int(ALL_NUM_DATA_C2)

ROW_WANTED_C1 = range(NUM_ROW_WANTED_C1)
ROW_WANTED_C2 = range(NUM_ROW_WANTED_C2)

ROW_WANTED_PRE_C1 = range(ALL_NUM_DATA_C1 - NUM_ROW_WANTED_C1, ALL_NUM_DATA_C1)
ROW_WANTED_PRE_C2 = range(ALL_NUM_DATA_C2 - NUM_ROW_WANTED_C2, ALL_NUM_DATA_C2)

COL_WANTED = range(ALL_NUM_FEATURE)

############################################data preprocessing
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
y_C1 = np.repeat(1, len(x_train_C1_normalized))
y_C2 = np.repeat(0, len(x_train_C2_normalized))
y_train = np.concatenate((y_C1, y_C2), axis = 0)


x_train_all_normalized = np.concatenate((x_train_C1_normalized, x_train_C2_normalized), axis = 0)

x_test_normalized = DP.normalization(x_test, mean_all, var_all)


col_1_train = np.repeat(1, len(x_train_all_normalized))
col_1_test = np.repeat(1, len(x_test_normalized))

col_1_train.shape = len(col_1_train), 1
col_1_test.shape = len(col_1_test), 1


x_train_all_normalized = np.concatenate((col_1_train, x_train_all_normalized), axis = 1)
x_test_normalized = np.concatenate((col_1_test, x_test_normalized), axis = 1)


w_0 = np.repeat(0, len(x_train_all_normalized[0]))
w_0.shape = 1, len(w_0)
ada = 0

L_set = np.array([])



#############################################gradient decent
for i in range(1000):
	w_0, ada = DP.Gradient_Decent_Logestic(w_0, LEARNING_RATE, x_train_all_normalized, y_train, ada)
	#L_set = np.append(L_set, DP.cross_entropy(w_0, x_train_all_normalized, y_train))

#plt.plot(L_set, "b")
#plt.show()

y_pre_C1 = np.array([])
df_w0 = pd.DataFrame(w_0)
df_w0.to_csv("W.csv", header = False, index = False)
z = np.dot(x_test_normalized, w_0.T)
print(z.shape)


print("done")
##########################################testing
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



df_y_pre.to_csv("Y_PRE_logestic_0.csv", index = True, header = ["label"], index_label = "id")
