from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
#splitting data in labels and features
X = iris.data
y = iris.target
print(X.shape, " ", y.shape)
#hours of study vs grades
#10 different students
#train with 8 students
#test with 2 students 
#allows to determine model accuracy
X_train , X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
