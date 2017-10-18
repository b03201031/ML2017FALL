import pandas as pd 
import numpy as np 
import math


def data_preprocessing(path_x, path_y):
	df_data_x = pd.read_csv(path_x)
	df_data_y = pd.read_csv(path_y)

	arr_type_1 = np.array([])
	arr_type_0 = np.array([])
	
	len_type_1 = 0
	len_type_0 = 0

	for label in df_data_y.values:

		if (label == 1): ##>50K
			arr_type_1 = np.append(arr_type_1, df_data_x.values[len_type_1+len_type_0, :])
			len_type_1 = len_type_1+1

		elif (label == 0):
			arr_type_0 = np.append(arr_type_0, df_data_x.values[len_type_1+len_type_0, :])
			len_type_0 = len_type_0+1



	arr_type_0.shape = len_type_0, int(len(arr_type_0)/len_type_0)
	arr_type_1.shape = len_type_1, int(len(arr_type_1)/len_type_1)
	
	df_data_type_0 = pd.DataFrame(arr_type_0, index = None, columns = df_data_x.columns)
	df_data_type_1 = pd.DataFrame(arr_type_1, index = None, columns = df_data_x.columns)

	df_data_type_1.to_csv("X_train_type_1.csv", index = None)
	df_data_type_0.to_csv("X_train_type_0.csv", index = None)


def get_data(path, col_wanted, row_wanted):
	df_data = pd.read_csv(path)
	arr_data = df_data.values

	out_arr = np.array([])

	for i in row_wanted:
		for j in col_wanted:
			out_arr = np.append(out_arr, arr_data[i, j])

	out_arr.shape = len(row_wanted), len(col_wanted)

	return out_arr


#x here is all of the data with first data in the first row
#it return a row vector for each feature
def get_mean(x):
	row_dim, col_dim = x.shape

	x_mean = np.array([])

	for j in range(row_dim):
		tmp_sum = 0
		for i in range(columns):
			tmp_sum = tmp_sum + x[i, j]

		x_mean = np.append(x_mean, tmp_sum/len(x))

	return x_mean


#x here is all of the data with first data in the first row
def get_covariance_matrix(x, x_mean):
	for i in len(x_mean):
		x[:, i] = x[:, i] - x_mean[i]

	cov = np.dot(np.transpose(x),x)/(len(x))

	return cov



#f(x) = (1/(2pi)^(D/2))*((1/det(sigma))^(1/2))*exp(-1/2*(tanspose(x-mean) dot inverse(sigma) dot (x-mean)))
# x is a row vector here so transpose need to replace by each other
def prob_Gaussian(x, mean, sigma):
	det_sigma = np.linalg.det(sigma)
	return (1/(2*math.pi)**(len(x)/2))*(math.sqrt((1/det_sigma)))*math.exp((-0.5)*np.dot(np.dot(x-mean, np.linalg.inv(sigma)), np.transpose(x-mean)))










