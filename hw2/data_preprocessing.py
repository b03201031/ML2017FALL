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


def get_data(path, col_beg, col_end, row_beg, row_end):
	df_data = pd.read_csv(path)
	arr_data = df_data.values

	return arr_data[row_beg:row_end, col_beg:col_end]


#x here is all of the data with first data in the first row
#it return a row vector for each feature
def get_mean(x):
	row_dim, col_dim = x.shape

	x_mean = np.array([])

	for j in range(col_dim):
		tmp_sum = np.sum(x[:, j])

		x_mean = np.append(x_mean, tmp_sum/len(x))

	return x_mean


#x here is all of the data with first data in the first row
def get_covariance_matrix(x, x_mean):
	for i in range(len(x_mean)):
		x[:, i] = x[:, i] - x_mean[i]

	cov = np.dot(np.transpose(x),x)/(len(x))

	return cov

def combinate_covariance(cov_1, cov_2, weight_1, weight_2):
	return (weight_1*cov_1 + weight_2*cov_2)/(weight_1+weight_2)


def prob_class(data_1, data_2):
	num_data_1 = len(data_1)
	num_data_2 = len(data_2)
	num_all_data = num_data_2 + num_data_1


	return num_data_1/num_all_data, num_data_2/num_all_data



#f(x) = (1/(2pi)^(D/2))*((1/det(sigma))^(1/2))*exp(-1/2*(tanspose(x-mean) dot inverse(sigma) dot (x-mean)))
# x is a row vector here so transpose need to replace by each other
def prob_Gaussian(x, mean, sigma):
	det_sigma = np.linalg.det(sigma)
	return (1/(2*math.pi)**(len(x)/2))*(math.sqrt((1/det_sigma)))*math.exp((-0.5)*np.dot(np.dot(x-mean, np.linalg.inv(sigma)), np.transpose(x-mean)))


#single data in c1
def prob_in_class_single(x, mean_c1, mean_c2, sigma_comb, prob_c1, prob_c2):

	return prob_Gaussian(x, mean_c1, sigma_comb)*prob_c1 / (prob_Gaussian(x, mean_c1, sigma_comb)*prob_c1+prob_Gaussian(x, mean_c2, sigma_comb)*prob_c2)

def accuracy(pred_y, real_y):
	correct = 0
	for i in range(len(pred_y)):
		if pred_y[i] == real_y[i]:
			correct = correct + 1


	return correct/len(pred_y)
	
def prob_in_class_multi(x, mean_c1, mean_c2, sigma_comb, prob_c1, prob_c2):

	y_pre = np.array([])

	for i in range(len(x)):
		tmp_prob = prob_in_class_single(x[i], mean_c1, mean_c2, sigma_comb, prob_c1, prob_c2)
		if tmp_prob >= 0.5:
			y_pre = np.append(y_pre, 1)
		elif tmp_prob < 0.5:
			y_pre = np.append(y_pre, 0)

	return y_pre




