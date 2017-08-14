import numpy as np
import matplotlib.pyplot as plt

def read_dataset(filePath):
    with open(filePath, 'r') as read:
        data = read.readlines()

    X = []
    Y = []
    for item in data:
        line = item.strip().split(',')
        X.append(float(line[0]))
        Y.append(float(line[1]))
    return X, Y

def h(theta, x):
    return (theta[0, 0] + theta[0, 1] * x)

def cost_function(train_data, y, m, theta):
    sum = 0
    for i in range(0, m):
        sum = sum + (h(theta, train_data[i, 1])-y[i])**2
    cost = (1/float(2*m))*sum
    return cost

def gradient_descent(train_data, y, m, alpha, iterations, theta):

    ret = []
    for j in range(0, iterations):
        sum0 = 0
        sum1 = 0
        H = []
        for i in range(0, m):
            sum0 = sum0 + ((h(theta, train_data[i, 1]) - y[i]) * train_data[i, 0])
            sum1 = sum1 + ((h(theta, train_data[i, 1]) - y[i]) * train_data[i, 1])
            H.append(h(theta, train_data[i, 1]))
        val0 = alpha * (1 / float(m)) * sum0
        val1 = alpha * (1 / float(m)) * sum1
        theta[0, 0] = theta[0, 0] - val0
        theta[0, 1] = theta[0, 1] - val1
        cost = cost_function(train_data, y, m, theta)
        print(cost)
        ret = H
    return ret
#main

x, y = read_dataset('data/ex1data1.txt')
lst = []
for item in x:
    lst.append([1, item])

train_x = np.array(lst)
theta = np.zeros((1, 2))
iterations = 1500
alpha = 0.01
print(cost_function(train_x, y, len(x), theta))
H = gradient_descent(train_x, y, len(x), alpha, iterations, theta)

#plotting data
plt.plot(train_x[:, 1], y, 'rx')
plt.plot(train_x[:, 1], H, 'b-')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('ex1')
plt.show()


