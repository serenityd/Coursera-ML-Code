import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def computeCost(X, Y, theta):
    m=len(Y)
    J=1/(2*m)*(X*theta-Y).T*(X*theta-Y)
    return J
def gradientDescent(X, Y, theta, alpha, iterations):
    for i in range(0,iterations):
        theta = theta - 1/m*alpha*X.T*(X*theta-Y)
    return theta
data=np.loadtxt('ex1data1.txt',delimiter=',')
X=np.mat((data[:,0]))
X.shape=-1,1
Y=np.mat(data[:,1])
Y.shape=-1,1
m=len(X)
plt.plot(X,Y,'ro')
print(np.mat(np.ones(m)).T.shape)
print(X.shape)
X=np.c_[np.mat(np.ones(m)).T,X]
print(X.shape)
theta=np.mat(np.zeros(2))
theta.shape=-1,1
iterations = 1500;
alpha = 0.01;
J = computeCost(X, Y, theta)
print('With theta = [0 ; 0]\nCost computed = %s\n'%float(J))
print('Expected cost value (approx) 32.07\n')
J = computeCost(X, Y,np.mat([[-3.6303],[1.1664]]))
print('\nWith theta = [-3.6303 ; 1.1664]\nCost computed = %f\n'%float(J))
print('Expected cost value (approx) 54.24\n');
theta = gradientDescent(X, Y, theta, alpha, iterations)
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')
plt.plot(X[:,1],X*theta,'-')
plt.show()
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
print(np.mat(np.r_[theta0_vals[0],theta1_vals[1]]).T.shape)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.mat(np.r_[theta0_vals[i],theta1_vals[j]]).T
        J_vals[i,j] = computeCost(X, Y, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
[theta0_vals, theta1_vals] = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, rstride=5, cstride=5, cmap=cm.Accent, linewidth=0.5)
plt.show()
counterfig=plt.figure()
ax=counterfig.add_subplot(111)
CS = ax.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2,3,20))
plt.clabel(CS, inline=1, fontsize=10)
ax.plot(theta[0,0], theta[1,0], 'rx', markersize=10, linewidth=2)
plt.show()