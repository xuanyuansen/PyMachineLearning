#coding=utf-8
'''
Created on 2013年11月2日

@author: Wangliaofan
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename='testSet.txt'):
    dataMat = []; labelMat = []
    #print sys.path[0]
    fr = open(filename,'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #size n*3
        labelMat.append(int(lineArr[2]))                            #size 1*n
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    n = shape(dataMatrix)[1]
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def gradNewton(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    middle = dataMatrix.transpose() * dataMatrix
    weights = middle.I * dataMatrix.transpose() * labelMat
    return weights

#Logistic Regression Classification
def gradDescent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix size 1*n
    n = shape(dataMatrix)[1]                 #size n*3
    alpha = 0.001
    eps=0.0001
    weights = ones((n,1))                   #size 3*1
    while True:                             #transpose,inverse
        h = sigmoid(dataMatrix*weights)     # n*3 X 3*1 = n*1
        error=(h.transpose()-labelMat.transpose())*dataMatrix      
                                            # 1*n X n*3 = 1*3
        suberror= alpha*error
        weights = weights - suberror.transpose()
        maxerror=0.0000001
        for element in suberror.flat:
            maxerror=max(maxerror,abs(element))
        if maxerror<eps:
            break
    return weights

#IRLS Method Logistic Regression Classification
def IRLSmethod(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose()
    n,m=shape(dataMatrix)                  
    R=zeros((n,n))
    Rnn=mat(R)
    eps=0.000001
    weights_old = mat(0.0001*ones((m,1)))
    iternum=0
    while True:
        #plotBestFit(weights_old)
        iternum=iternum+1
        Y= sigmoid(dataMatrix*weights_old)       
        for i in range(n):
            Rnn[i,i]=Y[i]*(1-Y[i])
        temp1= dataMatrix.transpose()*Rnn*dataMatrix
        temp2= dataMatrix.transpose()*Rnn
        RnnInverse=mat(zeros((n,n)))
        for iv in range(n):
            if 0==Rnn[iv,iv]:
                RnnInverse[iv,iv]=1.0
            else:
                RnnInverse[iv,iv]=1.0/Rnn[iv,iv]
        temp3= dataMatrix*weights_old - RnnInverse*(Y-labelMat)
        weights_new = temp1.I * temp2 * temp3
        suberror= weights_new - weights_old
        print suberror
        maxerror=0.0000000001
        for element in suberror.flat:
            maxerror=max(maxerror,abs(element))
            break
        #print iternum,weights_new,maxerror
        if maxerror<eps:
            break
        for k in range(m):
            weights_old[k] = weights_new[k]
        
    return weights_new

def plotBestFit(weights): 
    weight=weights.getA()
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def plotBothBestFit(weights,otherweights): 
    weight=weights.getA()
    otherweight=otherweights.getA()
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weight[0]-weight[1]*x)/weight[2]
    y2 = (-otherweight[0]-otherweight[1]*x)/otherweight[2]
    ax.plot(x, y,'r', x, y2,'g--')
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def calculate_error_rate(dataMat,labelMat,weights):
    weight=weights.getA()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    error_count=0 
    for i in range(n):
        y_temp=(-weight[0]-weight[1]*dataArr[i,1])/weight[2]
        if ( dataArr[i,2]>y_temp and int(labelMat[i])== 1 ) or (dataArr[i,2]<y_temp and int(labelMat[i])== 0):
            error_count = error_count+1
    return float(error_count)/float(n)

if __name__ == '__main__':
    dataArr,LabelMat=loadDataSet()
    #print dataArr,LabelMat
    #weights=gradAscent(dataArr,LabelMat)
    #otherweights=gradDescent(dataArr,LabelMat)
    #directweights=gradNewton(dataArr,LabelMat)
    newweights=IRLSmethod(dataArr,LabelMat)
    print newweights
    error_rate=calculate_error_rate(dataArr,LabelMat,newweights)
    print "error rate: ",error_rate
    plotBestFit(newweights)
    #print weights,otherweights,directweights,newweights
    #plotBestFit(weights)
    #plotBestFit(otherweights)
    #plotBothBestFit(weights,otherweights)
    #plotBestFit(directweights)
    pass