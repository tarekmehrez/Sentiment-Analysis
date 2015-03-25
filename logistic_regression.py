import numpy as np
import scipy.optimize as opt
import sys

def sigmoid(X):
	return 1 / (1 + np.exp(-X))



def cost(theta, X,y):
	hyp = sigmoid(np.dot(X, theta))

	cost = -y * np.log(hyp) - (1 - y) * np.log(1-hyp)
	return cost.mean()


def grad(theta, X, y):
    hyp = sigmoid(np.dot(X, theta))
    error = hyp - y 
    grad = np.dot(error, X) / y.size

    return grad

print "reading vectors"
with open("vectors-bag-of-words.csv") as file:
	file_content = np.array([line.strip().decode('utf-8').split(',') for line in file])

y = file_content[:,0] 
X = file_content[:,1:]

y = (y == 'positive') * 1
X = np.append( np.ones((X.shape[0], 1)), X, axis=1)

theta = 0.1*np.random.randn(X.shape[1])


X=np.array(X,dtype=float)
y=np.array(y,dtype=float)
theta=np.array(theta,dtype=float)

print "running logistic regression"
theta = opt.fmin_bfgs(cost, theta, fprime=grad, args=(X, y))

predictions = sigmoid(np.dot(X,theta))

print np.sum((predictions == y) * 1) / y.shape[0] * 100