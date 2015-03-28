import numpy as np
import logging



class LogisticRegression(object):

	def __init__(self,X, y,iterations,alpha,reg,percentage):
		training = int(X.shape[0] * percentage)
		self.X_train = X[0:training,:]
		self.X_test = X[training:,:]

		self.y_train = y[0:training]
		self.y_test = y[training:]
		self.iterations = iterations
		self.alpha = alpha
		self.theta = 0.1*np.random.randn(X.shape[1])
		self.reg = reg
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

	#  sigmoid function

	def sigmoid(self,X):
		return 1 / (1 + np.exp(-X))

	# cost function

	def cost(self,theta, X,y):
		hyp = self.sigmoid(np.dot(X, theta))

		cost = -y * np.log(hyp) - (1 - y) * np.log(1-hyp)
		return cost.mean() + (self.reg/(2 * X.shape[0]) * sum(np.power(theta,2)))


	# gradient descent

	def grad(self,theta, X, y):
		for i in range(self.iterations):
			hyp = self.sigmoid(np.dot(X, theta))
			error = hyp - y
			current_cost = self.cost(theta, X,y)

			if(current_cost == 0):
				self.logger.info("Cost reached 0")
				return theta

			theta_0 = theta[0]
			theta = theta - (self.alpha * (np.dot(error,X) + ((self.reg / X.shape[0]) * theta)))
			if self.reg > 0:
				theta[0] = theta_0
				

			self.logger.debug("Iteration: " + str(i) + ", Cost: " + str(current_cost))

		return theta

	def train(self):
		self.logger.info("Starting Logistic Regression")
		self.logger.debug("Number of training instances: " + str(self.X_train.shape[0]))
		self.logger.debug("Number of testing instances: " + str(self.X_test.shape[0]))
		theta = self.grad(self.theta, self.X_train, self.y_train)
		hyp = self.sigmoid(np.dot(self.X_test, self.theta))


		self.logger.debug("Done Training with Accuracy: "+str(((((hyp >= 0.5) * 1) == self.y_test) * 1).mean()* 100) + "%")
