import numpy as np
import pandas as pd
import math #Used for Pi and log()
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ones,vstack
import time

path = '/Users/kevindeangeli/Desktop/Fall2019/COSC522/Project2/Dataset_PimaTr.txt'
X = pd.read_csv(path, delim_whitespace = 1, header=None)
#X.columns=['x1', 'x2','x3','x4', 'x5', 'x6','x7', 'y']
#print(X.head(2))
print(" ")


#Task 1
X[X.shape[1]-1] = X[X.shape[1]-1].map({'Yes': 1, 'No': 0})
#X=X.drop(X.columns[0], axis=1)
print(" ")
#print(X.head(3))


path2 = '/Users/kevindeangeli/Desktop/Fall2019/COSC522/Project2/Dataset_PimaTest.txt'
X_test = pd.read_csv(path2, delim_whitespace = 1, header=None)
X_test[X_test.shape[1]-1] = X_test[X_test.shape[1]-1].map({'Yes': 1, 'No': 0})
#X_test=X_test.drop(X_test.columns[0], axis=1)
#print(" ")
#print(X_test.head(3))
#print(X.shape[1]) #Get the number of columns

#Obtaining means and Variances
meanArr = np.mean(X, axis=0)
#print("means", meanArr)
varArr = np.std(X, axis=0)
#print("std ", varArr)


#Normalizing Data:
#print(X.head(3))
nX=X
nX_test = X_test
for i in range(X.shape[1]-1):
    nX.loc[:,i] = (X.loc[:,i] - meanArr[i]) / varArr[i]
    nX_test.loc[:,i] = (X_test.loc[:,i] - meanArr[i]) / varArr[i]
#print(X.head(3))
#display(nX)

#print(X.columns)


def euclidian_dsitance(x, X):
    start_time = time.time()
    # x is the test point, X is the dataset
    # Using Euclidian Distance
    distancesArr = []
    # For each row in the data set:
    for index, row in X.iterrows():
        dist = 0
        # For each column-1 in the dataset:
        for i in range(X.shape[1] - 1):
            dist += (x[i] - X.iloc[index][i]) ** 2
        label = X.loc[index][X.shape[1] - 1]
        distanceAndLabel = (dist, label)
        distancesArr.append(distanceAndLabel)
    return distancesArr


def guessLabel(x, X, k):
    label0 = 0
    guessClass = 9
    label1 = 0
    ks = []
    distancesArr = euclidian_dsitance(x, X)
    distancesArr.sort()
    for p in range(int(k)):
        minimum = min(distancesArr)
        ks.append((minimum[0], minimum[1]))
        distancesArr.remove(minimum)
    for q in range(len(ks)):
        if ks[q][1] == 0:
            label0 += 1
        else:
            label1 += 1
    if label0 >= label1:
        guessClass = 0
    else:
        guessClass = 1
    return guessClass
    # print("------------------------")
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print("Closest neighboords and respective labels: ", ks)
    # print("Guess label", guessClass)
    # print("------------------------")


def knnAccuracy_KArray(testX, X, k):
    # Here testX is the test data set
    # X is the trainning data set
    # K is the number of neighbors
    # let label 1 represent True
    # and Label 0 represent False
    start_time = time.time()
    accuracyList = []
    TP, TN, FN, FP = 0, 0, 0, 0
    for index, row in testX.iterrows():
        trueClass = testX.loc[index][testX.shape[1] - 1]
        testPoint = testX.loc[index, 0:X.shape[1] - 2]
        predictedClass = guessLabel(testPoint, X, k)
        if trueClass == predictedClass:
            if trueClass == 1:
                TP += 1
            else:
                TN += 1
        else:
            if trueClass == 0:
                FP += 1  # Since it was guess as 1.
            else:
                FN += 1
    accuracy = (TP + TN) / testX.shape[0]
    accuracyList.append(accuracy)
    accuracyArray = np.array(accuracyList)
    print("TP: ", TP, "TN: ", TN, "FP: ", FP, "FN: ", FN)
    print("accuracy: ", accuracy * 100)
    print("------------------------")
    print("--- %s seconds ---" % (time.time() - start_time))
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    ax.plot(k, accuracyArray)
    ax.set(xlabel='Prior Probabbility (W)', ylabel='Accuracy',
           title=' ')
    # title='Finding the best accuracy')
    ax.grid()
    # plt.show()


knnAccuracy_KArray(nX_test, nX, 1)