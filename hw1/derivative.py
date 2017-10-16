import numpy as np
import math

#y is PM2.5 of prediction hour
#x is training data
# beta_0 ~ beta_n
# x y is np array
# x: [x11...]	y:row vector
#	 [x21...]...

def Gradient(x, y, beta, lamda):
	
	NUM_DATA_SET = len(x)

	#transform y into column vecotor
	col_y = np.reshape(y, (np.size(y), 1))

	 #transform beta into column vector
	col_beta = np.reshape(beta, (np.size(beta), 1))

	###calculate###
	
	#x dot beta => [b1x1+b2x2...] (transfor into row vec)
	col_y_0 = np.dot(x, col_beta)

	#[y - (b0+b1x1...)]
	L = col_y - col_y_0 

	#[sum(y^i - (b0+b1x1+...)^i(x1^i)) , ...]*-2 i is idx of row of x
	x_trans = np.transpose(x)
	gra = np.dot(x_trans, L)*-2

	tmp_beta = beta
	for i in range(1,len(beta)):
		tmp_beta[i] = tmp_beta[i]*2*lamda

	output = np.transpose(gra) + tmp_beta
	return output



#output theta_1
#theta_1 = theta_0 - learning_rate * gradient(L(theta_0))
def Gradient_Decent(theta_0, learning_rate, x, y, ada, k, lamda):
	g = Gradient(x, y,theta_0, lamda)
	ada = np.sqrt((ada**2 + g*g))
	theta_1 = theta_0 - learning_rate*g/ada
	return theta_1, ada





#single data
def Regression(beta, x):

	col_beta = np.reshape(beta, (np.size(beta), 1))

	y = np.dot(x,col_beta)
	y.shape = np.size(y)

	return y

def Loss(beta, x, y, lamda):
	
	col_beta = np.reshape(beta, (np.size(beta), 1))
	col_y = np.reshape(y, (np.size(y),1))
	M = col_y - np.dot(x, col_beta)
	L = M*M

	tmp_beta = beta**2

	output = np.sum(L) + lamda*np.sum(tmp_beta[1:])


	return output


def ERROR(x_test, y_test, beta):
	error = 0
	for i, j in zip(x_test, y_test):
		error = error + abs(j - Regression(beta, i))
	return error

def DATA_PREPROCESSING(all_data, num_data_set, duration, IDX_PM25, month, data_selected):
	NUM_DATA_MONTH = 480
	MAX_MONTH = len(all_data[0, :]) / NUM_DATA_MONTH
	MAX_NUM_DATA_SET = NUM_DATA_MONTH - NUM_DATA_MONTH%(duration+1) 

	if (month > MAX_MONTH):
		print("MONTH EXCEEDED")
		return

	if (num_data_set > MAX_NUM_DATA_SET):
		print("num_data_set EXCEEDED")
		return

	x = np.array([])
	y = np.array([])
	for i in range((month-1)*480, (month-1)*480+num_data_set):
		for j in data_selected:
			x = np.concatenate((x, np.reshape(all_data[j, i:i+duration], duration)), axis = 0)
		
		y = np.append(y, np.reshape(all_data[IDX_PM25, i+duration], 1), axis = 0)

	x.shape = num_data_set, duration*len(data_selected)

	return x, y

def RMSE(x_test, y_test, beta):
	
	rmse = 0
	idx = 0

	for i in x_test:
		rmse = rmse + (Regression(beta, i) - y_test[idx]) ** 2
		idx = idx+1


	rmse = math.sqrt(rmse / len(y_test))
	return rmse

def Mean(x):
	#print(x)
	sum_x = np.array([])

	for  i in range(len(x[0, :])):
		tmp_sum = np.sum(x[:, i])
		
	
		if (i == 0):
			sum_x = tmp_sum
			continue

		sum_x = np.append(sum_x, tmp_sum)
		
	
	mean = sum_x / len(x)


	return mean

def SD(x):
	mean_x = Mean(x)
	sd = np.array([])
	for i in range(len(x[0, :])):
		tmp_sum = 0
		for j in range(len(x)):
			tmp_sum = tmp_sum + (x[j, i]-mean_x[i])**2

		tmp_sd = math.sqrt(tmp_sum/len(x))

		if i == 0:
			sd = tmp_sd
			continue

		sd = np.append(sd,tmp_sd)


	return sd