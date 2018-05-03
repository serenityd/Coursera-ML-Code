import numpy as np
import matplotlib.pyplot as plt
def featureNormalize(X):
    mu= np.mean(X,axis=0)
    sigma= np.std(X,axis=0)
    X=(X-mu)/sigma
    return [X,mu,sigma]
def computeCost(X, Y, theta):
    m=len(Y)
    J=1/(2*m)*(X*theta-Y).T*(X*theta-Y)
    return J
def Leastsquare(X,Y):
    theta=(X.T*X).I*X.T*Y
    return theta
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history=np.zeros(num_iters)
    for i in range(0,num_iters):
        theta = theta - 1/m * alpha * X.T * ( X *theta -y)
        J_history[i]=computeCost(X,y,theta)
    return [theta,J_history]
data=np.loadtxt('ex1data2.txt',delimiter=',')
X=np.mat((data[:,[0,1]]))
Y=np.mat(data[:,-1]).T
m=len(X)
[X,mu,sigma] = featureNormalize(X)
X=np.c_[np.ones((len(X),1)),X]
print(X)
alpha = 0.01
num_iters = 400
theta = np.mat(np.zeros((3,1)))
[theta, J_history] = gradientDescentMulti(X, Y, theta, alpha, num_iters)
print(theta)
cost=computeCost(X,Y,theta)
print(cost)
plt.plot(J_history,color = 'b')
plt.xlabel('iters')
plt.ylabel('j_cost')
plt.title('cost variety')
plt.show()
predict=[1650,3]
predict=(predict-mu)/sigma
pre_X=np.c_[1,predict]
pre_Y=pre_X*theta
theta=Leastsquare(X,Y)
print(theta)
cost=computeCost(X,Y,theta)
print(cost)
predict=[1650,3]
predict=(predict-mu)/sigma
pre_X=np.c_[1,predict]
pre_Y=pre_X*theta