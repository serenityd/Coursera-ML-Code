import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def computeCost(X, Y, theta):
    m=len(Y)
    J=1/(2*m)*(X*theta-Y).T*(X*theta-Y)
    return J
def Leastsquare(X,Y):
    theta=(X.T*X).I*X.T*Y
    return theta
data=np.loadtxt('ex1data1.txt',delimiter=',')
X=np.mat((data[:,0]))
X.shape=-1,1
Y=np.mat(data[:,1])
Y.shape=-1,1
m=len(X)
plt.plot(X,Y,'ro')
X=np.c_[np.mat(np.ones(m)).T,X]
theta=np.mat(np.zeros(2))
theta.shape=-1,1
theta = Leastsquare(X,Y)
print(theta)
plt.plot(X[:,1],X*theta,'-')
plt.show()
print(computeCost(X,Y,theta))
