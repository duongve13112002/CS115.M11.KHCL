import numpy as np 
import pandas as pd 
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

data = pd.read_csv('pima-indians-diabetes.data.csv',header = None)
#print(data)

X_Train = (data.iloc[:,0:8].values).astype('float64')
y_train = data.iloc[:,[8]].values
x_train = scale(X_Train)
#print(x_train)
print('----------')

model = LogisticRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_train)
print(y_predict)


print(accuracy_score(y_train,y_predict))

x1 = np.array([X_Train[i] for i in range(len(y_train)) if y_predict[i] == 1])
x0 = np.array([X_Train[i] for i in range(len(y_train)) if y_predict[i] == 0])




