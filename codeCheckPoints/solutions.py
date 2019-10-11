import numpy as np
import pandas as pd
import math #Used for Pi and log()
import sympy as sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ones,vstack
import time

def load_data(f):
    #f is the route of the data file.
    x = pd.read_csv(f, delim_whitespace=1, header=None)
    x[x.shape[1] - 1] = x[x.shape[1] - 1].map({'Yes': 1, 'No': 0})
    #Could return x if I wanted to keep it as the first project.
    X = x.loc[:,0:x.shape[1]-2]
    Y = x.loc[:,x.shape[1]-1]
    Y=Y.to_numpy()
    X=X.to_numpy()
    X=normalization(X)
    return X, Y

def normalization(X):
    meanArr = np.mean(X, axis=0)
    varArr = np.std(X, axis=0)
    nX = X[:]
    for i in range(X.shape[1]):
        nX[:, i] = (X[:, i] - meanArr[i]) / varArr[i]
    return nX

def pca(nX,percentEerror=.9,showGraph=False):
    nX_Cov = np.cov(nX.T) #Note that covariannce in pd is calculated differently
    nX_eig, nX_eigV = np.linalg.eig(nX_Cov)
    ordered_eigs = -np.sort(-nX_eig)
    totalSum = np.sum(ordered_eigs)
    # Store the indexes of the ordered eigenvalues
    # So you can create a matrix of eigenvectors
    # it the correct order:
    order_index_eigs = []
    for i in range(ordered_eigs.shape[0]):
        order_index_eigs.append(np.where(nX_eig == ordered_eigs[i])[0].item())


    # Store the indexes of the ordered eigenvalues
    # So you can create a matrix of eigenvectors
    # it the correct order:
    order_index_ordered_eigs = []
    for i in range(ordered_eigs.shape[0]):
        order_index_ordered_eigs.append(np.where(nX_eig == ordered_eigs[i])[0].item())


    def eigenValErrorAnalysis(ordered_eigs):
        plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
        x = np.linspace(1, ordered_eigs.shape[0], ordered_eigs.shape[0])
        # Cumulative:
        eg = []
        totalEig = 0
        for i in range(ordered_eigs.shape[0]):
            totalEig += ordered_eigs[i]
            eg.append(totalEig / totalSum)
        plt.plot(x, eg)
        plt.xlabel(xlabel='Number of eigenvalues')
        plt.ylabel(ylabel='Percent of Information Explained')
        plt.show()

    if showGraph == True:
        eigenValErrorAnalysis(ordered_eigs)

    def numberofEigValsToUse(eigs,percentEerror):
        tmp = 0
        ix = -1
        P = []
        for i in range(eigs.shape[0]):
            tmp += eigs[i] / totalSum
            if tmp > percentEerror:
                ix = i + 1  # Number of eigenvectors to use is 5
                for k in range(ix):
                    a = nX_eigV[order_index_eigs[k]]
                    P.append(a)
                P = np.array(P)
                return P

    P = numberofEigValsToUse(ordered_eigs,percentEerror)
    return np.dot(nX,np.transpose(P))

def fld(nX,y=0, training= True):
    if training:
        #split the data int two classes:
        #key = y[:, 0] == 0
        key0 = y==0
        y0Values = nX[key0]
        key1 = y == 1
        y1Values= nX[key1]

        y0ValuesMean = np.mean(y0Values,axis=0)
        y1ValuesMean = np.mean(y1Values,axis=0)


        y0Cov = np.cov(y0Values.T)
        y1Cov = np.cov(y1Values.T)

        S_0 = (y0Values.shape[0] - 1) * y0Cov
        S_1 = (y0Values.shape[0] - 1) * y1Cov

        S_w = S_0 + S_1
        S_w_inv = np.linalg.inv(np.array(S_w))

        fld.v = np.dot(S_w_inv, (np.transpose(y0ValuesMean) - np.transpose(y1ValuesMean)))


        y0Values=np.dot(y0Values, fld.v)
        y1Values=np.dot(y1Values, fld.v)


        nX[:, 0][key0] = y0Values
        nX[:, 0][key1] = y1Values
        nX=np.delete(nX, np.s_[1:nX.shape[1]], axis=1)
    else:
        print(fld.v)
        nX = np.dot(nX, fld.v)
    return nX




class Knn:
    def __init__(self):
        self.nX = []
        self.pX = []
        self.fX = []
        self.predictionArr =[]
        self.totalTime= -1

    def showTime(self):
        print("Time in seconds: ", self.totalTime)

    def fit(self, X, y):
        #self.nX = normalization(X)
        self.nX = X
        self.pX = pca(self.nX)
        self.fX = fld(self.nX,y)
        self.y = y

    #x here is just a point
    def euclidian_dsitanceList(self, x, X):
        # x is the test point, X is the dataset
        # Using Euclidian Distance
        distancesArr = []
        # For each row in the data set:
        dist = -1
        index=0 # used to associate a distance with a y.
        for row in X:
            testPoint = row[0:X.shape[1]]
            dist = np.linalg.norm(testPoint - x)
            labelIndex = self.y[index]
            distanceAndLabel = (dist, labelIndex)
            distancesArr.append(distanceAndLabel)
            index +=1
        return distancesArr


    #x here is just a point
    def guessLabel(self,x, X, k):
        label0 = 0
        guessClass = -1
        label1 = 0
        ks = []
        distancesArr = self.euclidian_dsitanceList(x, X)
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


    def knn(self,XTest, X, k):
        #X is the training data
        guessList =[]
        guessedLabel = -1
        #print(XTest)
        for row in XTest:
            guessedLabel = self.guessLabel(row, X, k)
            guessList.append(guessedLabel)
        return guessList



    def predict(self, XTest, k=1, data="nX"):
        start_time = time.time()
        predictionArr =[]
        #print(XTest)
        if data == "nX":
            self.predictionArr = self.knn(XTest, self.nX, k)
        elif data == "pX":
            XTest = pca(XTest)
            self.predictionArr = self.knn(XTest, self.pX, k)
        else:
            XTest=fld(XTest, training=False)
            self.predictionArr = self.knn(XTest, self.fX, k)
        #print(self.predictionArr)
        self.totalTime= time.time() - start_time
        return  self.predictionArr


def accuracy_score(yTest, y_model):
    TP, TN, FP, FN = 0, 0, 0, 0
    index=0
    for rightLabel in yTest:
        guessLabel=y_model[index]
        index+=1
        if guessLabel == rightLabel:
            if rightLabel == 1:
                TP += 1
            else:
                TN += 1
        else:
            if rightLabel == 1:
                FN += 1
            else:
                FP += 1
    # print(TP)
    totalRowsInData = yTest.shape[0]
    confusion_matrix = [['TP', TP], ["TN", TN], ['FP', FP], ['FN', FN], ['Accuracy', (TN + TP) / totalRowsInData]]
    confusionarr = [TP, TN, FP, FN]
    print(confusion_matrix)
    return confusion_matrix, confusionarr










def main():
    trainingData=  '/Users/kevindeangeli/Desktop/Fall2019/COSC522/Project2/Dataset_PimaTr.txt'
    testingData= '/Users/kevindeangeli/Desktop/Fall2019/COSC522/Project2/Dataset_PimaTest.txt'
    Xtrain,Ytrain = load_data(trainingData)
    Xtest,ytest = load_data(testingData)
    model = Knn()
    model.fit(Xtrain, Ytrain)

    #Predicts accepts values "nX", "pX", "fX". "nX" is by default.
    y_model = model.predict(Xtest,k=21, data="fX")
    accuracy = accuracy_score(ytest, y_model)


    #print('accuracy = ', accuracy)


if __name__ == "__main__":
    main()