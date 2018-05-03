import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
def plotData(X,y):
    #对多维数组拷贝，并降维
    pos=X[np.where(y==1,True,False).flatten()]
    neg = X[np.where(y == 0, True, False).flatten()]
    plt.plot(pos[:,0],pos[:,1],'o')
    plt.plot(neg[:, 0], neg[:, 1], 'x')

def sigmoid(z):
    h=1/(1+np.exp(-z))
    return h
def costFunction(initial_theta, X, y):
    initial_theta=np.mat(initial_theta)
    initial_theta=initial_theta.T
    m=len(y)
    cost=1/m*(-y.T*np.log(sigmoid(X*initial_theta))-(1-y.T)*np.log(1-sigmoid(X*initial_theta)))
    initial_theta = initial_theta.T
    return cost
def gradientFunction(initial_theta, X, y):
    initial_theta=np.mat(initial_theta)
    m = len(y)
    initial_theta=initial_theta.T
    gradiant=1/m*X.T*(sigmoid(X*initial_theta)-y)
    initial_theta = initial_theta.T
    return gradiant.T
data=np.loadtxt(fname='ex2data1.txt',delimiter=',')
print(data.shape)
X = np.mat(data[:,[0,1]])
y = np.mat(data[:,-1]).T
m=len(X)
print(X.shape,y.shape,)
plotData(X, y)
plt.legend(['Admitted','Not admitted'],loc='upper right')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
X=np.c_[np.ones(m),X]
initial_theta=np.mat(np.zeros(3))
print(initial_theta.shape)
cost = costFunction(initial_theta, X, y)
print(cost)
gradiant=gradientFunction(initial_theta, X, y)
print(gradiant)
test_theta = np.mat([-24,0.2,0.2])
cost = costFunction(test_theta, X, y)
print(cost)
gradiant=gradientFunction(test_theta, X, y)
print(gradiant)
print(initial_theta.shape)
res = minimize(fun=costFunction,x0=initial_theta, method='TNC',jac=gradientFunction, args=(X, y))

theta = res.x
cost = res.fun

# Print theta to screen
print('Cost at theta found by scipy: %f' % cost)
print('theta:', ["%0.4f" % i for i in theta])



