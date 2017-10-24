import data_preprocessing as DP 
import numpy as np

x = np.array([[1,2], [2,4]])
x_mean = DP.get_mean(x)
x_cov = DP.get_covariance_matrix(x, x_mean)
#print(np.linalg.det(sigma*2*math.pi))