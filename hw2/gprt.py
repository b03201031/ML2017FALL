from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
X_PATH = "X_train"
NUM_DATA_X = 32561

Y_PATH = "Y_train"
X_TEST_PATH = "X_test"

NUM_TRAIN = 32000
NUM_TEST = 32561-32000

df_x_train = pd.read_csv(X_PATH)
x_train =  df_x_train.values[:NUM_TRAIN, :]
#x_test = df_x_train.values[NUM_TRAIN:, :]

df_y_train = pd.read_csv(Y_PATH)
y_train = df_y_train.values[:NUM_TRAIN, :]
#y_test = df_y_train.values[NUM_TRAIN:, :]


df_x_test = pd.read_csv(X_TEST_PATH)
x_test = df_x_test.values

clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 2, random_state = 0).fit(x_train, y_train)


y_pre = clf.predict(x_test)

idx = []
for i in range(len(y_pre)):
	idx.append(i+1)
df_y_pre = pd.DataFrame(y_pre, index = idx)
df_y_pre.to_csv("Y_PRE_gprt_04.csv", index = True, header = ["label"], index_label = "id")