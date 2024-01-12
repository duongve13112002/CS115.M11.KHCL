x = [1, 2, 3, 4, 5, 6, 7]
y = [5, 9, 13, 17, 21, 25, 29]
w = 3
b = 1
Learning_rate = 0.001
dai = len(x)
def Gardient_w():
    s = 0
    for i in range(dai):
        s += (y[i] - w*x[i] - b) * (-x[i])
    return w - Learning_rate * (s/dai)
def Gardient_b():
    s = 0
    for i in range(dai):
        s += (y[i] - w*x[i] - b) * -1
    return b - Learning_rate * (s/dai)
def Gardient_loss():
    l = 0
    for i in range(dai):
        l += (y[i] - w*x[i] - b)**2
    return l / (2*dai)

for i in range(10000):
    w = Gardient_w()
    b = Gardient_b()
    l = Gardient_loss()
    if l < 10 **(-5):
        break  
print('w: ' + str(w))
print('b: '+str(b) , end ="\n"+'l: '+str(l))
