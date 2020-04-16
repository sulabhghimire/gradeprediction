import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data = pd.read_csv("Dataset/student-mat.csv", sep=';')

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
#print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
X = preprocessing.scale(X)

#print(X)
Y = np.array(data[predict])
#print(Y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.1)
#trying to get a higher value model as answer
'''best = 0
for i in range(1000) :
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, train_size=0.1)
    #print(x_train)
    #print(y_train)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > best :
        with open('predict_grade_using_modules.pickle', 'wb') as f:
            pickle.dump(linear, f)
        best = acc
'''
pickle_in = open('predict_grade_using_modules.pickle', 'rb')
linear = pickle.load(pickle_in)

print('Co : \n', linear.coef_)
print('Intercept : ', linear.intercept_)

predictions = linear.predict(x_test)

''' 
p = 'absences'
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()
'''