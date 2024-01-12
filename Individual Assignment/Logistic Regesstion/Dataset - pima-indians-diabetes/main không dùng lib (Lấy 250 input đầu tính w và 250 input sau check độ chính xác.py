import numpy as np 
import pandas as pd 

def max_rows(X):
	max_list_row = []
	for i in range(0,len(X[0])):
		max_row = X[0][i]
		for j in range(0,len(X)):
			if X[j][i] > max_row:
				max_row = X[j][i]
		max_list_row.append(max_row)
	return max_list_row			

data = pd.read_csv('pima-indians-diabetes.data.csv',header = None)
#print(data)

#X1 là những output = 1
X1 = data.loc[data[8] == 1]
#print(X1)
#X0 là những output =0 
X0 = data.loc[data[8] == 0]
#X1,X0 chỉ lấy cột dữ liệu ( tính thứ 0)
X1 = X1.iloc[:,0:8].values
X0 = X0.iloc[:,0:8].values

#Scale dữ liệu
#Lấy dữ liệu  input output
X =(data.iloc[:,0:8].head(250).values).astype('float64')
Y = data.iloc[:,[8]].head(250).values
max_list_row = max_rows(X)
#Chuẩn hóa dữ liệu thành dạng từ (-1,1) tại mỗi cột thì tao lấy giá trị lớn nhất xong lấy từng giá tri trong cột đó chia cho MAX 
for i in range(0,len(X)):
	for j in range(0,len(X[i])):
		X[i][j] = X[i][j] / max_list_row[j]
#print(X[0])

#Hàm sigmoid
def sigmoid(x):
	return 1/(1+np.exp(-x))

#thêm cột 1 vào
ones = np.ones((len(X),1))
x = np.concatenate((ones,X),axis = 1)
w = np.array([0.,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]).reshape(-1,1)
print('----------')

#Lặp
numOFIteration = 10000
cost = np.zeros((numOFIteration,1))
learning_rate = 0.0005 
numloop = []
for i in range(0,numOFIteration):
	#Tính giá trị dự đoán
	y_predict = sigmoid(np.dot(x,w))
	cost[i] = -np.sum(np.multiply(Y,np.log(y_predict)) + np.multiply(1-Y,np.log(1-y_predict)))
	#Gardient descent
	w = w - learning_rate*np.dot(x.T,y_predict-Y)
	numloop.append(i)
numloop = np.array(numloop)
#Sai số khởi đầu
print('Sai số khởi điểm: ' + str(cost[0]))
#sai số cuối cùng
print('Sai số cuối cùng: ' + str(cost[999]))
print('----------')
#trọng số cuối cùng
print('Trọng số cuối cùng: ')
print(w)
print('Lost function: ')



#Độ chính xác đối với 250 giá trị cuối 
a = 0.00

X_tail =(data.iloc[:,0:8].tail(250).values).astype('float64')
Y_tail = data.iloc[:,[8]].tail(250).values
max_list_row_tail = max_rows(X_tail)
for i in range(0,len(X_tail)):
	for j in range(0,len(X_tail[i])):
		X_tail[i][j] = X_tail[i][j] / max_list_row_tail[j]
x_tail = np.concatenate((ones,X_tail),axis = 1)
y_predict_tail = sigmoid(np.dot(x_tail,w))
for i in range(0,len(y_predict_tail)):
	if y_predict_tail[i] <0.5:
		y_predict_tail[i] = 0
	else:
		y_predict_tail[i] = 1
	if y_predict_tail[i] == Y_tail[i]:
		a+= 1
print('Chính xác: '+str((a/len(Y))*100)+ '%')		

#Hiển thị dữ liệu sau khi phân chia
a1 = []
a0 = []
for i in range(0,len(y_predict)):
	if y_predict[i] == 1:
		a1.append(X[i])
	else:
		a0.append(X[i])
z1 = np.array(a1)
z0 = np.array(a0)



#Nhập dữ liệu cần dự đoán 
input = [[6.00,148.00,72.00,35.00,0.00,33.60,0.627,50.00]]
need_pre = np.array(input)
print('Input:',input[0:])
for i in range(0,np.size(need_pre,1)):
	need_pre[0,i] /= max_list_row[i]
#Dự đoán
ones = np.ones((len(need_pre),1))
need_pre = np.concatenate((ones,need_pre),axis  = 1)
result = sigmoid(np.dot(need_pre,w))
print(result)
if result >= 0.5:
	print('=> Kết quả: 1')
else:
	print('=> Kết quả: 0')

