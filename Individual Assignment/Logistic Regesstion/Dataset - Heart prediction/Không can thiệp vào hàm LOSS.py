import numpy as np 
import pandas as pd
from sklearn.model_selection import  train_test_split #Dùng hàm này để phân ra các train,test 1 cách dê dàng 

def max_rows(X):
	max_list_row = []
	for i in range(0,len(X[0])):
		max_row = X[0][i]
		for j in range(0,len(X)):
			if X[j][i] > max_row:
				max_row = X[j][i]
		max_list_row.append(max_row)
	return max_list_row			
def min_rows(X):
	min_list_row = []
	for i in range(0,len(X[0])):
		min_row = X[0][i]
		for j in range(0,len(X)):
			if X[j][i] < min_row:
				min_row = X[j][i]
		min_list_row.append(min_row)
	return min_list_row	

missing_value = "?"#Liệt kê những kí tự mất để có thể thay thế
#Đọc file và thay thế những giá trị bị mất = 0
data_1 = pd.read_csv('processed.cleveland.data',sep = ",",header = None,na_values= missing_value,dtype = np.float64).fillna(0)
data_2 = pd.read_csv('reprocessed.hungarian.data',sep = " ",header = None,na_values= missing_value,dtype = np.float64).fillna(0)


#Lọc ra các input và output
X1 = data_1.iloc[:,:-1].values 
X2 = data_2.iloc[:,:-1].values
Y1 = data_1.iloc[:,-1].values
Y2 = data_2.iloc[:,-1].values
#Nối các input và output lại 
X = np.concatenate((X1,X2),axis = 0)
Y = np.concatenate((Y1,Y2),axis = 0)
#Chỉnh lại những input nào khác 0 là bằng 1
for i in range (0,len(Y)):
	if Y[i] != 0:
	 	Y[i] = 1
#Lấy ra các dữ liệu train và test	 	
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Chuẩn hóa dữ liệu train và test
max_list_row = max_rows(x_train)
min_list_row = min_rows(x_train)
for i in range(0,len(x_train)):
	for j in range(0,len(x_train[i])):
		x_train[i][j] = (x_train[i][j] - min_list_row[j])/ (max_list_row[j] -min_list_row[j])

for i in range(0,len(x_test)):
	for j in range(0,len(x_test[i])):
		x_test[i][j] = (x_test[i][j] - min_list_row[j])/ (max_list_row[j] -min_list_row[j])

#Hàm sigmoid
def sigmoid(x):
	return 1/(1+np.exp(-x))

#Bổ dung cột toàn giá trị 1 và input
ones = np.ones((len(x_train),1))
x = np.concatenate((ones,x_train),axis = 1)

#Khỏi tạo ra w 1 cách ngẫu nhiên
w =[0]
for i in range(0,len(x[0])-1):
	w.append(0.1)
w = np.array(w).reshape(-1,1)

y_train = np.array(y_train).reshape(-1,1)
numOFIteration = 10000
cost = np.zeros((numOFIteration,1))
learning_rate = 0.0001 
numloop = []
for i in range(0,numOFIteration):
	#Tính giá trị dự đoán
	y_predict = sigmoid(np.dot(x,w))
	cost[i] = -np.sum(np.multiply(y_train,np.log(y_predict)) + np.multiply(1-y_train,np.log(1-y_predict)))
	#Gardient descent
	w = w - learning_rate*np.dot(x.T,y_predict-y_train)
	numloop.append(i)
numloop = np.array(numloop)
print('Sai số khởi điểm: ' + str(cost[0]))
#sai số cuối cùng
print('Sai số cuối cùng: ' + str(cost[9999]))
print('----------')
#trọng số cuối cùng
print('Trọng số cuối cùng: ')
print(w)
#print('Lost function: ')


ones = np.ones((len(x_test),1))
x_test_ne = np.concatenate((ones,x_test),axis = 1)
y_predict_test = sigmoid(np.dot(x_test_ne,w))

#Tính toán các Evaluation metrics
a_value_1 = 0.00
a_value_0 = 0.00
a_false_0 = 0.00
a_false_1 = 0.00
for i in range(0,len(y_predict_test)):
	if y_predict_test[i] <0.5:
		y_predict_test[i] = 0
		if(y_test[i] == 1):
			a_false_0 +=1			
	else:
		y_predict_test[i] = 1
		if(y_test[i] == 0):
			a_false_1 += 1			
	if y_predict_test[i] == y_test[i] and y_test[i] == 1:
		a_value_1+= 1
	if y_predict_test[i] == y_test[i] and  y_test[i] == 0:
		a_value_0+= 1
#print('Chính xác với output là 0: '+str((a_value_0/len(Y))*100)+ '%')	
#print('Chính xác với output là 1: '+str((a_value_1/len(Y))*100)+ '%')
print(a_false_0) #false negative DU doan la 0 nhung dap an la 1
print(a_false_1) # flase possitive DU doan la 1 nhung dap an la 0
print(a_value_0)
print(a_value_1)
print('-------------------------------------------')
print('Confusion Matrix\t Actually Positive(1)\t Actually Negative(0)')
print('Predict Positive(1) \t'+ str((a_value_1/len(x_test))*100),'\t\t\t\t '+str((a_false_1/len(x_test))*100))
print('Predict Negative(0) \t'+str((a_false_0/len(x_test))*100),'\t\t\t\t '+str((a_value_0/len(x_test))*100))		
accuracy_true = (a_value_1 + a_value_0)/(a_value_1 + a_value_0+a_false_0+a_false_1)
preccission = a_value_1/(a_false_1+a_value_1)
recall = a_value_1/(a_value_1+a_false_0)
F1 = 2*((preccission*recall)/(preccission+recall))
print('accuracy_true:',accuracy_true*100,'%')
print('preccission:',preccission*100,'%')
print('recall:',recall*100,'%')
print('F1:',F1*100,'%')
	