import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def euclidean_distance(a, b):
    distance = 0
    a = np.array(a)
    b = np.array(b)
    for d in range(len(b)):
        distance += abs(a[d] - b[d]) ** 2

    distance = distance ** (1 / 2)

    return distance
if __name__ == '__main__':
    print('\n-------------- Q 2.1 ------------------------')

    data = pd.read_table('D2z.txt', sep=" ", header=None, names=["X1", "X2", "Y"])

    # Split Features and target
    X, y = data.drop([data.columns[-1]], axis=1), data[data.columns[-1]]

    testdata = {'X1': [], 'X2': [], 'Y': []}
    f1 = np.arange(-2.0, 2.1, 0.1)
    f2 = np.arange(-2.0, 2.1, 0.1)
    for x1 in f1:
        for x2 in f2:
            testdata['X1'].append(x1)
            testdata['X2'].append(x2)
            testdata['Y'].append(-1)

    test = pd.DataFrame(testdata)
    testX = test.drop([test.columns[-1]], axis=1)
    for testIndex in testX.index:
        distances = []
        for index in X.index:
            distances.append(euclidean_distance(testX.iloc[testIndex], X.iloc[index]))
        minpos = distances.index(min(distances))
        test.at[testIndex, "Y"] = data.iloc[minpos]["Y"]
        # print(test.iloc[testIndex])

    plt.scatter(test['X1'], test['X2'], c=test['Y'],alpha=0.1)
    plt.scatter(data['X1'], data['X2'], c=data['Y'], marker='x')
    plt.show()
