import numpy as np
import matplotlib.pyplot as plt

class regularizedLogisticRegression(object):
    def __init__(self, filePath, learningRate, epochs, lamda):
        self.__filePath = filePath
        self.__trainX, self.__trainY = [], []
        self.__learning_rate, self.__epochs, self.__lamda = learningRate, epochs, lamda

    def read_dataset(self):
        with open(self.__filePath, 'r') as read:
            data = read.readlines()
        X, Y = [], []
        for item in data:
            line = item.strip().split(',')
            #line[0], line[1] = float(line[0])*0.01, float(line[1])*0.01
            X.append(list(map(lambda v: float(v), [line[0], line[1]])))
            Y.append(float(line[2]))
        self.__trainX, self.__trainY = np.array(X), np.array(Y)
        self.__theta = np.zeros(1 + self.__trainX.shape[1])
        self.__X, self.__Y = np.array(X), np.array(Y)
        #self.mean_normalization()

    def fit(self):
        self.cost = []
        for i in range(0, self.__epochs):
            net_input = self.calc(self.__trainX)
            h = self.sigmoid(net_input)
            error = h - self.__trainY
            grad = self.__trainX.T.dot(error)
            grad += (self.__lamda/float(len(self.__trainX))) * self.__theta[1:]
            self.__theta[1:] -= self.__learning_rate * (grad)
            self.__theta[0] -= self.__learning_rate * error.sum()
            self.cost.append(self.cost_function(h))


    def cost_function(self, h):
        m = len(self.__trainY)
        cost, t = 0, 0
        for i in range(0, m):
            cost += (-self.__trainY[i]*np.log(h[i]) - (1 - self.__trainY[i])*np.log(1 - h[i]))
        for i in range(1, len(self.__theta)):
            t += (self.__theta[i]**2)
        return (1/float(m)) * cost + ((self.__lamda/float(2*m)) * t)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calc(self, X):
        return np.dot(X, self.__theta[1:]) + self.__theta[0]

    def predict(self, test):
        z = self.calc(test)
        return np.where(self.sigmoid(z) >= 0.5, 1, 0)


    def get_accuracy(self):
        result = self.predict(self.__trainX)
        print(result)
        print(self.__trainY)
        cnt = 0
        for i in range(0, len(result)):
            if result[i] == self.__trainY[i]:
                cnt = cnt + 1
        return (cnt / len(result)) * 100

    def mean_normalization(self):
        avg1 = self.__trainX[:, 0].sum() / float(len(self.__trainX))
        avg2 = self.__trainX[:, 1].sum() / float(len(self.__trainX))
        min1, max1 = self.__trainX[:, 0].min(), self.__trainX[:, 0].max()
        min2, max2 = self.__trainX[:, 1].min(), self.__trainX[:, 1].max()
        for i in range(0, len(self.__trainX)):
            self.__trainX[i, 0] = (self.__trainX[i, 0] - avg1) / float(max1 - min1)
            self.__trainX[i, 1] = (self.__trainX[i, 1] - avg2) / float(max2 - min2)

    def getX(self, v, p):
        X = []
        for i in range(0, len(self.__trainX)):
            if self.__Y[i] == v:
                X.append(self.__X[i, p])
        return np.array(X)


#main
rlr = regularizedLogisticRegression("data/ex2data2.txt", 0.001, 500, 0)
rlr.read_dataset()
rlr.fit()
print(rlr.get_accuracy())
plt.plot(rlr.getX(1, 0), rlr.getX(1, 1), 'rx')
plt.plot(rlr.getX(0, 0), rlr.getX(0, 1), 'bo')

plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('ex2')
plt.show()
