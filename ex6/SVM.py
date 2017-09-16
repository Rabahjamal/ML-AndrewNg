import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


class SVM(object):
    def __init__(self, filePath, sigma = 1):
        self.__filePath = filePath
        self.__mat, self.__sigma = scipy.io.loadmat(self.__filePath), sigma
        self.__trainX, self.__y = self.__mat['X'], np.ravel(self.__mat['y'], order='C')


    def linearClassification(self):
        clf = svm.SVC(C = 1, kernel='linear')
        clf.fit(self.__trainX, self.__y)
        self.plotBoundary(clf,0,4.5,1.5,5)

    def plotBoundary(self, my_svm, xmin, xmax, ymin, ymax):
        xvals = np.linspace(xmin, xmax, 100)
        yvals = np.linspace(ymin, ymax, 100)
        zvals = np.zeros((len(xvals), len(yvals)))
        for i in range(len(xvals)):
            for j in range(len(yvals)):
                zvals[i][j] = float(my_svm.predict(np.array([[xvals[i], yvals[j]]])))

        zvals = zvals.transpose()
        u, v = np.meshgrid(xvals, yvals)
        mycontour = plt.contour(xvals, yvals, zvals, [0])
        plt.title("Decision Boundary")

    def nonLinearClassification(self):
        clf = svm.SVC(C=0.3, kernel='rbf', gamma=self.__sigma ** -2)
        clf.fit(self.__trainX, self.__y)
        self.plotBoundary(clf,0,1,.4,1.0)

    def computeGaussianKernel(self):
        K = np.zeros((self.__trainX.shape[0], self.__trainX.shape[0]))
        for i in range(0, len(self.__trainX)):
            f = np.zeros((self.__trainX.shape[0],))
            for j in range(0, len(self.__trainX)):
                x1 = self.__trainX[i, 0] - self.__trainX[j, 0]
                x2 = self.__trainX[i, 1] - self.__trainX[j, 1]
                f[j] = (np.exp(-(x1**2 + x2**2)/float(2*self.__sigma**2)))
            K[i] = f
        return K

    def test(self):
        self.__Xval, self.__yval = self.__mat['Xval'], self.__mat['yval']
        search = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
        bestScore, bestC, bestSigma = 0, 0, 0
        for C in range(0, len(search)):
            for sigma in range(0, len(search)):
                clf = svm.SVC(C=search[C], kernel='rbf', gamma=search[sigma] ** -2)
                clf.fit(self.__trainX, self.__y)
                score = clf.score(self.__Xval, self.__yval)
                if score > bestScore:
                    bestScore, bestC, bestSigma = score, search[C], search[sigma]
        clf = svm.SVC(C=bestC, kernel='rbf', gamma=bestSigma ** -2)
        clf.fit(self.__trainX, self.__y)
        self.plotBoundary(clf, -.5,.3,-.8,.6)



    def getPoints(self):
        pos = np.array([self.__trainX[i] for i in range(self.__trainX.shape[0]) if self.__y[i] == 1])
        neg = np.array([self.__trainX[i] for i in range(self.__trainX.shape[0]) if self.__y[i] == 0])
        return pos, neg



#main

#example dataset1

ex6_1 = SVM('data/ex6data1.mat')
ex6_1.linearClassification()
pos, neg = ex6_1.getPoints()

plt.plot(pos[:, 0], pos[:, 1], 'r+')
plt.plot(neg[:, 0], neg[:, 1], 'bo')
plt.show()

#example dataset2
ex6_2 = SVM('data/ex6data2.mat', 0.1)
ex6_2.nonLinearClassification()
pos, neg = ex6_2.getPoints()

plt.plot(pos[:, 0], pos[:, 1], 'r+')
plt.plot(neg[:, 0], neg[:, 1], 'bo')
plt.show()

#example dataset3
ex6_3 = SVM('data/ex6data3.mat', 0.1)
ex6_3.test()

pos, neg = ex6_3.getPoints()

plt.plot(pos[:, 0], pos[:, 1], 'r+')
plt.plot(neg[:, 0], neg[:, 1], 'bo')
plt.show()
