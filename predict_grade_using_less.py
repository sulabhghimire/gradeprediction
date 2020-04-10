import pandas as pd
import numpy as np

data = pd.read_csv('Dataset/student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'absences', 'failures', 'studytime', 'health']]
#print(data.head())

predict = 'G3'

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(X)-25):
    x_train.append(X[i])
    y_train.append(Y[i])

for i in range(len(X)-25, len(X)):
    x_test.append(X[i])
    y_test.append(Y[i])

learning_rate = 0.001
w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
w4 = np.random.randn()
w5 = np.random.randn()
w6 = np.random.randn()
b = np.random.randn()

# training set
for i in range(9000000):
    ri = np.random.randint(len(x_train))
    point = x_train[ri]

    pred = point[0] * w1 + point[1] * w2 + point[2] * w3 + point[3] * w4 + point[4] * w5 + point[5] * w6 + b
    target = y_train[ri]

    cost = (pred - target) ** 2

    dcost_dpred = 2 * (pred - target)

    dpred_dw1 = point[0]
    dpred_dw2 = point[1]
    dpred_dw3 = point[2]
    dpred_dw4 = point[3]
    dpred_dw5 = point[4]
    dpred_dw6 = point[5]
    dpred_db = 1

    dcost_dw1 = dcost_dpred * dpred_dw1
    dcost_dw2 = dcost_dpred * dpred_dw2
    dcost_dw3 = dcost_dpred * dpred_dw3
    dcost_dw4 = dcost_dpred * dpred_dw4
    dcost_dw5 = dcost_dpred * dpred_dw5
    dcost_dw6 = dcost_dpred * dpred_dw6
    dcost_db = dcost_dpred * dpred_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    w3 = w3 - learning_rate * dcost_dw3
    w4 = w4 - learning_rate * dcost_dw4
    w5 = w5 - learning_rate * dcost_dw5
    w6 = w6 - learning_rate * dcost_dw6
    b = b - learning_rate * dcost_db

for i in range(len(x_test)):
    point = x_test[i]

    pred = point[0] * w1 + point[1] * w2 + point[2] * w3 + point[3] * w4 + point[4] * w5 + point[5] * w6 + b

    print("Input {}".format(point))
    print("Our Output : {}".format(pred))
    print("Expected Output : {}".format(y_test[i]))




