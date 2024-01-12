from csv import reader
import numpy as np 


def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset




def Loadcsv(path):
    try:
        raw = load_csv(path)
        raw = np.array(raw,dtype = np.float64)
        X = np.zeros((np.size(raw,0),np.size(raw,1)))
        X[:,0] = 1
        X[:,1:] = raw[:,:-1]
        y = raw[:,-1]
        yield X
        yield y
    except:
        return 0


def predict(X,thelta):
    return X@ thelta

def computeCost(X,y,Theta):
    check = predict(X,Theta)-y
    m = np.size(y)
    j = (1/(2*m))*np.transpose(check) @ check 
    return j

def GradientDescent(X,y,alpha=0.02,iter=5000): 
    theta = np.zeros(np.size(X,1)) 
    J_hist = np.zeros((iter,2)) 
    m = np.size(y)
    X_T = np.transpose(X)
    pre_cost = computeCost(X,y,theta)
    for i in range(0,iter):
        error = predict(X,theta) - y
        theta = theta - (alpha/m)*(X_T @ error)
        cost = computeCost(X,y,theta)
        if np.round(cost,15) == np.round(pre_cost,15):
            print('Giá trị cực đại tại  I = %d ; J = %.6f'%(i,cost))
            J_hist[i:,0] = range(i,iter)
            J_hist[i:,1] = cost
            break
        pre_cost = cost
        J_hist[i, 0] = i
        J_hist[i, 1] = cost
    yield theta
    yield J_hist

def Normalize(X):
    n = np.copy(X)
    n[0,0] = 100
    s = np.std(n,0, dtype = np.float64)
    uy = np.mean(n,0)
    n = (n-uy)/s
    n[:,0] = 1
    yield n
    yield uy
    yield s
