import numpy as np 
from functions import *

filename = 'winequality-white.csv'
[X, y] = Loadcsv(filename)
[X, mu, s] = Normalize(X)
[Theta, J_hist] = GradientDescent(X,y,0.05,50000)
#Đưa input check output
'''input = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
input = (input-mu)/s
input[0] = 1
predict = predict(input,Theta)
print('%.15f'%(predict))
print(Theta)'''

