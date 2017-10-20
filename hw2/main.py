import data_preprocessing as DP


##C1 is type 1 here
PATH_X_TRAIN_C1 = "X_train_type_1.csv"
PATH_X_TRAIN_C2 = "X_train_type_0.csv"

ALL_NUM_DATA_C1 = 7841
ALL_NUM_DATA_C2 = 24720
ALL_NUM_FEATURE = 106


NUM_ROW_WANTED_C1 = int(ALL_NUM_DATA_C1*0.9)
NUM_ROW_WANTED_C2 = int(ALL_NUM_DATA_C2*0.9)

ROW_WANTED_C1 = range(NUM_ROW_WANTED_C1)
ROW_WANTED_C2 = range(NUM_ROW_WANTED_C2)

ROW_WANTED_PRE_C1 = range(ALL_NUM_DATA_C1 - NUM_ROW_WANTED_C1, ALL_NUM_DATA_C1)
ROW_WANTED_PRE_C2 = range(ALL_NUM_DATA_C2 - NUM_ROW_WANTED_C2, ALL_NUM_DATA_C2)

COL_WANTED = range(ALL_NUM_FEATURE)

x_train_C1 = DP.get_data(PATH_X_TRAIN_C1, 0, ALL_NUM_FEATURE-1, 0, NUM_ROW_WANTED_C1)
x_train_C2 = DP.get_data(PATH_X_TRAIN_C2, 0, ALL_NUM_FEATURE-1, 0, NUM_ROW_WANTED_C2)

x_test_C1 = DP.get_data(PATH_X_TRAIN_C1, 0, ALL_NUM_FEATURE-1, ALL_NUM_DATA_C1 - NUM_ROW_WANTED_C1, ALL_NUM_DATA_C1-1)
x_test_C2 = DP.get_data(PATH_X_TRAIN_C2, 0, ALL_NUM_FEATURE-1, ALL_NUM_DATA_C2 - NUM_ROW_WANTED_C2, ALL_NUM_DATA_C2-1)


mean_C1 = DP.get_mean(x_train_C1)
mean_C2 = DP.get_mean(x_train_C2)


sigma = DP.combinate_covariance(DP.get_covariance_matrix(x_train_C1, mean_C1), DP.get_covariance_matrix(x_train_C2, mean_C2), len(x_train_C1), len(x_train_C2))

prob_C1 = NUM_ROW_WANTED_C1/(NUM_ROW_WANTED_C1+NUM_ROW_WANTED_C2)
prob_C2 = NUM_ROW_WANTED_C2/(NUM_ROW_WANTED_C1+NUM_ROW_WANTED_C2)

print(DP.prob_Gaussian(x_train_C1[0], mean_C1, sigma))


'''
y_pre_C2 = DP.prob_in_class_multi(x_test_C2, mean_C1, mean_C2, sigma, prob_C1, prob_C2)
y_real_C2 = np.repeat(0, len(ROW_WANTED_PRE_C2))

print(DP.accuracy(y_pre_C2, y_real_C2))
'''