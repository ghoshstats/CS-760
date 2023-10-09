import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import warnings
from collections import Counter

warnings.filterwarnings("ignore")


def euclidean_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


def knn(data, test, threshold):
    testX, testY = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    X, y = data.drop([data.columns[-1]], axis=1), data[data.columns[-1]]
    X = X.iloc[:, 1:].to_numpy()
    y = y.to_numpy()
    testX = testX.iloc[:, 1:].to_numpy()
    testY = testY.to_numpy()
    ypred = []
    data = data.to_numpy
    k = 5
    for testIndex, test in enumerate(testX):
        distances = [euclidean_distance(test, x) for index, x in enumerate(X)]
        idx = np.argpartition(distances, k - 1)[:k]
        counter = Counter(y[idx])
        fraction = 0
        if counter.get(1) != None:
            fraction = (float)(counter.get(1)) / k
        if (fraction >= threshold):
            ypred.append(1)
        else:
            ypred.append(0)
        if (testIndex % 100 == 0):
            print(testIndex)
        # print(ypred)
    count = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    correct = 0
    for count in range(np.size(ypred)):
        if (ypred[count] == testY[count]):
            correct += 1
            if (testY[count] == 1):
                tp += 1
        if ((ypred[count] == 1) and (testY[count] == 0)):
            fp += 1
        if ((ypred[count] == 0) and (testY[count] == 1)):
            fn += 1
        if ((ypred[count] == 0) and (testY[count] == 0)):
            tn += 1
        count = count + 1
    accuracy = (float(correct) / float(count)) * 100
    recall = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return recall, fpr


class LogisticRegression():
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        for i in range(self.iterations):
            self.weights_update()
        return self

    def weights_update(self):
        A = 1 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        temp = (A - self.Y.T)
        temp = np.reshape(temp, self.m)
        dW = np.dot(self.X.T, temp) / self.m
        db = np.sum(temp) / self.m

        self.W = self.W - self.alpha * dW
        self.b = self.b - self.alpha * db

        return self

    def predict(self, X, threshold):
        P = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(P >= threshold, 1, 0)
        return Y


def main():
    data = pd.read_csv('emails.csv', sep=',')
    test = data.iloc[4000:5000, :]
    train = data.drop(test.index)

    X_train = train.iloc[:, 1:-1].values
    X_test = test.iloc[:, 1:-1].values
    Y_train = train.iloc[:, -1:].values
    testY = test.iloc[:, -1:].values.ravel()

    model = LogisticRegression(alpha=0.01, iterations=3000)

    model.fit(X_train, Y_train)
    ltpr = []
    lfpr = []
    for threshold in np.arange(0, 1.1, 0.1):
        ypred = model.predict(X_test, threshold)
        count = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        correct = 0
        # print(testY.shape)
        # print(ypred.shape)
        for count in range(np.size(ypred)):
            if (ypred[count] == testY[count]):
                correct += 1
                if (testY[count] == 1):
                    tp += 1
            if ((ypred[count] == 1) and (testY[count] == 0)):
                fp += 1
            if ((ypred[count] == 0) and (testY[count] == 1)):
                fn += 1
            if ((ypred[count] == 0) and (testY[count] == 0)):
                tn += 1
            count = count + 1
        accuracy = (float(correct) / float(count)) * 100
        recall = tp / (tp + fn)
        ltpr.append(recall)
        lfpr.append(fp / (tn + fp))
        print(ltpr)
        print(lfpr)
    test = data.iloc[4000:5000, :]
    train = data.drop(test.index)
    ktpr = []
    kfpr = []
    for threshold in np.arange(0, 1.1, 0.1):
        a, b = knn(train, test, threshold)
        ktpr.append(a)
        kfpr.append(b)
    print(ktpr)
    print(kfpr)
    plt.plot(kfpr, ktpr, color='blue')
    plt.plot(lfpr, ltpr, color='orange')

    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("False Positive Rate (Positive label: 1)")
    plt.legend(['KNeighborsClassifier', 'LogisticRegression'], loc=4)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

