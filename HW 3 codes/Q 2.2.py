import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import multiprocessing

def euclidean_distance(a, b):
    a=np.array(a)
    b=np.array(b)
    return np.linalg.norm(a-b)

def knn(data,test,fold):
  testX, testY = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
  X, y = data.drop([data.columns[-1]], axis=1), data[data.columns[-1]]
  X=X.iloc[:,1:].to_numpy()
  y=y.to_numpy()
  testX=testX.iloc[:,1:].to_numpy()
  testY=testY.to_numpy()
  ypred=[]
  data=data.to_numpy
  for testIndex,test in enumerate(testX):
      distances = [euclidean_distance(test, x) for index,x in enumerate(X)]
      minpos = distances.index(min(distances))
      ypred.append(y[minpos])
      if(testIndex%100==0):
       print(testIndex)

  count =0 
  tp=0
  fp=0
  fn=0
  correct=0  
  for count in range(np.size(ypred)):
    if(ypred[count]==testY[count]):
      correct+=1
      if(testY[count]==1):
        tp+=1
    if((ypred[count]==1) and (testY[count]==0)):
      fp+=1
    if((ypred[count]==0) and (testY[count]==1)):
      fn+=1
    count=count+1               
  accuracy = (float(correct)/float(count))*100
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  print ("\nAccuracy for {} : {}".format(fold,accuracy))
  print ("\nPrecision for {} : {}".format(fold,precision))
  print ("\nRecall for {} : {}".format(fold,recall))
  print(round(float(sum(ypred == testY)) / float(len(testY)) * 100, 2))



if __name__ == '__main__':#
    print('\n------------- Q 2.2 ------------------------')

    data = pd.read_csv('emails.csv',sep=',')
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      knn(train,test,i)

list(range(0,5000,1000))
