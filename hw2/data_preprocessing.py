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


def combinate_variance(var_1, var_2, weight_1, weight_2):
	return (weight_1*var_1 + weight_2*var_2)/(weight_1+weight_2)


def prob_class(data_1, data_2):
	num_data_1 = len(data_1)
	num_data_2 = len(data_2)
	num_all_data = num_data_2 + num_data_1


	return num_data_1/num_all_data, num_data_2/num_all_data



#f(x) = (1/((2pi)^(D/2)))*((1/det(sigma))^(1/2))*exp(-1/2*(tanspose(x-mean) dot inverse(sigma) dot (x-mean)))
# x is a rowvector with n featur 
#we return a row vector {P(Xn|Ck)}
def prob_Gaussian(x, mean, variance):

	prob_x = np.array([])
	for i in range(len(x)):
		
		if variance[i] == 0:
			prob_x = np.append(prob_x, 1)
			continue

		gauss = math.exp(-((x[i]-mean[i])**2)/(2*variance[i]))/(math.sqrt(2*math.pi*variance[i]))
		prob_x = np.append(prob_x ,np.clip(gauss,  0.0000000001, 0.9999999999))
	

	return prob_x


#single data in c1
def prob_in_class_single(x, mean_c1, mean_c2, var_comb, prob_c1, prob_c2):
	prob_set_c1 = prob_Gaussian(x, mean_c1, var_comb)
	prob_set_c2 = prob_Gaussian(x, mean_c2, var_comb)
	prob_posterior = (prob_set_c1*prob_c1)/(prob_set_c1*prob_c1+prob_set_c2*prob_c2)
	
	#return np.clip(np.prod(prob_posterior), 0.0000000000001, 0.9999999999999)
	return prob_posterior

def accuracy(pred_y, real_y):
	correct = 0
	for i in range(len(pred_y)):
		if pred_y[i] == real_y[i]:
			correct = correct + 1


	return correct/len(pred_y)
	
def prob_in_class_multi(x, mean_c1, mean_c2, sigma_comb, prob_c1, prob_c2):

	y_pre = np.array([])

	for i in range(len(x)):
		tmp_prob_c1 = prob_in_class_single(x[i, :], mean_c1, mean_c2, sigma_comb, prob_c1, prob_c2)

		y_pre = np.append(y_pre, tmp_prob_c1)
		'''
		tmp_prob_c2 = prob_in_class_single(x[i], mean_c2, mean_c1, sigma_comb, prob_c2, prob_c1)
		if tmp_prob_c1 >= tmp_prob_c2:
			y_pre = np.append(y_pre, 1)
		elif tmp_prob_c1 < tmp_prob_c2:
			y_pre = np.append(y_pre, 0)
		'''
	return y_pre



def get_w(mean_c1, mean_c2, sigma):
	return np.dot((mean_c1.T-mean_c2.T), np.linalg.inv(sigma))

def get_b(mean_c1, mean_c2, sigma, N1, N2):
	inv_sigma = np.linalg.inv(sigma)
	part_1 = (-0.5)*np.dot(mean_c1, inv_sigma)
	part_1 = np.dot(part_1, mean_c1.T) 

	part_2 = (0.5)*np.dot(mean_c2, inv_sigma)
	part_2 = np.dot(part_2, mean_c2.T) 

	part_3 = np.log(float(N1)/N2)

	return part_1+part_2+part_3
def get_z(w, b, x):
	return np.dot(w, x.T)+b

def sigmoid(z):

	
	return np.clip(1/(1.0+np.exp(-z)), 0.0000000000001, 0.9999999999999)



def predict(x_test, mean_c1, mean_c2, sigma, N1, N2):
	inv_sigma = np.linalg.inv(sigma)
	w = np.dot((mean_c1-mean_c2), inv_sigma)
	b = (-0.5)* np.dot(np.dot(mean_c1, inv_sigma), mean_c1.T)+ (0.5) * np.dot(np.dot(mean_c2, inv_sigma), mean_c2.T)+np.log(float(N1)/N2)
	x = x_test.T
	z = np.dot(w, x) + b
	y = sigmoid(z)

	return y

def normalization(x_data, mean, variance):
	num_row, num_col = x_data.shape
	x_normalized = np.zeros((num_row, num_col))
	for j in range(num_col):

		if (variance[j] != 0):
			x_normalized[:, j] = (x_data[:, j] - mean[j])/(np.sqrt(variance[j]))
		elif (variance[j] == 0):
			x_normalized[:, j] = (x_data[:, j] - mean[j])

	return	x_normalized


def Gradient(x, y, beta):
	
	NUM_DATA_SET = len(x)

	#transform y into column vecotor
	col_y = y.T

	 #transform beta into column vector
	col_beta = beta.T

	###calculate###
	
	#x dot beta => [b1x1+b2x2...] (z)
	col_z_0 = np.dot(x, col_beta)

	y_0 = np.array([])

	for z in col_z_0:
		if sigmoid(z) < 0.5:
			y_0 = np.append(y_0, 0)
		else:
			y_0 = np.append(y_0, 1)

	col_y_0 = y_0.T
	#[y - (b0+b1x1...)]
	L = col_y - col_y_0 

	#[sum(y^i - (b0+b1x1+...)^i(x1^i)) , ...]*-2 i is idx of row of x
	x_trans = x.T
	gra = np.dot(x_trans, L)*-1

	output = gra.T ##trans into row vec
	return output


def Gradient_Decent_Logestic(theta_0, learning_rate, x, y, ada):
	g = Gradient(x, y,theta_0)
	ada = np.sqrt((ada**2 + g*g))
	theta_1 = theta_0 - learning_rate*g/ada
	return theta_1, ada


def cross_entropy(w, x, y):
	f = np.dot(x, w.T)
	p_1 = y
	p_0 = 1 - y
	q_1 = f
	q_0 = 1 - f
	output = - np.sum(p_1*np.log(q_1)+p_0*np.log(q_0))
	return output