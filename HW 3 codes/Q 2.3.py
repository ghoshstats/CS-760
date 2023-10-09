import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


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

    def predict(self, X):
        P = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(P > 0.5, 1, 0)
        return Y


def main():
    data = pd.read_csv('emails.csv', sep=',')
    for i in range(0, 5000, 1000):
        test = data.iloc[i:i + 1000, :]
        train = data.drop(test.index)

        X_train = train.iloc[:, 1:-1].values
        X_test = test.iloc[:, 1:-1].values
        Y_train = train.iloc[:, -1:].values
        testY = test.iloc[:, -1:].values.ravel()

        model = LogisticRegression(alpha=0.01, iterations=3000)

        model.fit(X_train, Y_train)

        ypred = model.predict(X_test)
        count = 0
        tp = 0
        fp = 0
        fn = 0
        correct = 0
        print(testY.shape)
        print(ypred.shape)
        for count in range(np.size(ypred)):
            if (ypred[count] == testY[count]):
                correct += 1
                if (testY[count] == 1):
                    tp += 1
            if ((ypred[count] == 1) and (testY[count] == 0)):
                fp += 1
            if ((ypred[count] == 0) and (testY[count] == 1)):
                fn += 1
            count = count + 1
        accuracy = (float(correct) / float(count)) * 100
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print("\nAccuracy for {} : {}".format(i, accuracy))
        print("\nPrecision for {} : {}".format(i, precision))
        print("\nRecall for {} : {}".format(i, recall))
        print(round(float(sum(ypred == testY)) / float(len(testY)) * 100, 2))

#if __name__ == "__main__":
main()
