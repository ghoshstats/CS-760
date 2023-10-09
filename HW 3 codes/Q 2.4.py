import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import Counter

def euclidean_distance(a, b):
    a=np.array(a)
    b=np.array(b)
    return np.linalg.norm(a-b)

def knn(data,test,fold,k):
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
      idx = np.argpartition(distances, k-1)[:k]
      counter = Counter(y[idx])
      ypred.append(counter.most_common()[0][0])
      #print(ypred)
      if(testIndex%100==0):
       print(testIndex)
      #print(ypred)
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
  
  return correct



if __name__ == '__main__':#
    print('\n-------------- Q 2.4 ------------------------')

    data = pd.read_csv('emails.csv',sep=',')

    print("K=1")
    accuracy1=0
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      accuracy1=accuracy1+knn(train,test,i,1)
    accuracy1=(float(accuracy1)/5000.0)*100 

    print("K=3")
    accuracy3=0
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      accuracy3=accuracy3+knn(train,test,i,3)
    accuracy3=(float(accuracy3)/5000.0)*100 

    print("K=5")
    accuracy5=0
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      accuracy5=accuracy5+knn(train,test,i,5)
    accuracy5=(float(accuracy5)/5000.0)*100

    print("K=7")
    accuracy7=0
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      accuracy7=accuracy7+knn(train,test,i,7)
    accuracy7=(float(accuracy7)/5000.0)*100  
    
    print("K=10")
    accuracy10=0
    for i in range(0,5000,1000):
      test = data.iloc[i:i + 1000, :]
      train = data.drop(test.index)
      accuracy10=accuracy10+knn(train,test,i,10)
    accuracy10=(float(accuracy10)/5000.0)*100  
    print(accuracy10)
    K = [1,3,5,7,10]
    accuracy=[accuracy1,accuracy3,accuracy5,accuracy7,accuracy10]
    K=np.array(K)
    accuracy=np.array(accuracy)
    print(accuracy)
    plt.plot(K,accuracy)
    plt.show()
