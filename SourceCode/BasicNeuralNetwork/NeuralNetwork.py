#coding=utf-8
'''
Created on 2013年12月7日

@author: Wangliaofan
'''
#输入节点为i
#隐含节点为j
#输出节点为k
#一般情况下j>i
#使用后向传播算法，需要求出Wkj,Wji，从右向左。
#前向传播时系数可以表达为Wij，Wjk。
#最简单的情况，输入为2维，3个隐含节点，2种输出，采用sigmoid函数。
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename='testSet.txt'):
    dataMat = []; labelMat = []
    #print sys.path[0]
    fr = open(filename,'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])      #size m*2
        labelMat.append(int(lineArr[2]))                            #size 1*m
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1.0+exp(-inX))

def tangenth(inX):
    return (1.0*exp(inX)-1.0*exp(-inX))/(1.0*exp(inX)+1.0*exp(-inX))

def difsigmoid(inX):
    return sigmoid(inX)*(1.0-sigmoid(inX))

def BPTrainNetwork(dataMatIn,classLabels,i=2,j=3,k=1,maxCycles = 100000):
    W1=mat(zeros((j,i)))
    W2=mat(zeros((k,j)))
    for jj in range(j):
        W1[jj,:]=random.rand(i)
    for kk in range(k):
        W2[kk,:]=random.rand(j)
    print W1,W2
    aj=mat(zeros((j,1)))
    zj=mat(zeros((j,1)))
    yk=mat(zeros((k,1)))
    thetak=mat(zeros((k,1)))
    thetaj=mat(zeros((j,1)))
    backerror221=mat(zeros((j,1)))

    etha=0.00005
    eps=0.05
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels)             #convert to NumPy matrix size 1*m
    
    m = shape(dataMatrix)[0]
    
    labelMatnew = mat(zeros((2,m)))
    for idxm in range(m):
        labelMatnew[0,idxm]=labelMat[0,idxm]
        labelMatnew[1,idxm]= 1.0 - labelMat[0,idxm]
    
    errorfirstlayer=mat(zeros((j,i)))
    errorsecondlayer=mat(zeros((k,j)))
    print m
    
    errorsum_old=1000000.0
    cycle=0
    while True: 
        cycle += 1
        errorsum=0  
        for currentidx in range(m):
            #aj = sum W1ji*xi over i 
            for row in range(j):
                for col in range(i):
                    aj[row,0] += W1[row,col] * dataMatrix[currentidx,col]
                #zj[row,0]=sigmoid(aj[row,0])
                zj[row,0]=tangenth(aj[row,0])
                aj[row,0]=0
            #yk = sum W2kj*zj over j
            for ykiter in range(k):
                for row in range(j):
                    yk[ykiter,0] += W2[ykiter,row]*zj[row,0]
                thetak[ykiter,0] = sigmoid(yk[ykiter,0]) - labelMat[0,currentidx]
                #thetak[ykiter,0] = yk[ykiter,0] - labelMatnew[ykiter,currentidx]
                yk[ykiter,0]=0
                #errorsum += abs(thetak[ykiter,0])
                errorsum += 0.5 * thetak[ykiter,0] * thetak[ykiter,0]
            #thetaj = diff(Zj) * [sum (W2kj*thetak) over k]
            for row in range(j):
                for ykiter in range(k):
                    backerror221[row,0] += W2[ykiter,row] * thetak[ykiter,0]
                thetaj[row,0] = (1.0-zj[row,0]*zj[row,0])*backerror221[row,0]
                backerror221[row,0]=0
            #diffError(ji)=thetaj*xi,对所有的训练样本求错误的和
            for row in range(j):   
                for col in range(i):
                    errorfirstlayer[row,col] += thetaj[row,0] * dataMatrix[currentidx,col]
            #diffError(kj)=thetak*zj,对所有的训练样本求错误的和
            for ykiter in range(k):
                for row in range(j):
                    errorsecondlayer[ykiter,row] += thetak[ykiter,0] * zj[row,0]
            
        #更新系数
        for row in range(j):   
            for col in range(i):
                W1[row,col] = W1[row,col] - etha * errorfirstlayer[row,col]
                errorfirstlayer[row,col]=0

        for ykiter in range(k):
            for row in range(j):
                W2[ykiter,row] = W2[ykiter,row] - etha * errorsecondlayer[ykiter,row]
                errorsecondlayer[ykiter,row]=0
        
        print errorsum,cycle
        #print W1,W2
        if errorsum < eps or cycle >= maxCycles:
            break
        if errorsum > errorsum_old:
            print "发散了，冏"
            break
        errorsum_old = errorsum
        errorsum=0.0
        
        
    print 'iter num', cycle   
    return W1,W2

def NetworkPredict(dataMatIn,classLabels,W1ji,W2kj,i=2,j=3,k=1):
    dataMatrix=mat(dataMatIn)
    labelMat = mat(classLabels) 
    m=shape(dataMatrix)[0]
    aj=mat(zeros((1,j)))
    zj=mat(zeros((1,j)))
    Predict=mat(zeros((1,m)))
    rightClassify=0.0
    for idx_m in range(m):
        for idx_j in range(j):
            aj[0,idx_j]=0
            for idx_i in range(i):
                aj[0,idx_j] += W1ji[idx_j,idx_i]*dataMatrix[idx_m,idx_i]
            zj[0,idx_j]=tangenth(aj[0,idx_j])   
            
        for idx_k in range(k):
            temp=0
            for idx_j in range(j):  
                temp += W2kj[idx_k,idx_j]*zj[0,idx_j]
            Predict[0,idx_m]=sigmoid(temp)
               
    for idx_m in range(m):
        if (Predict[0,idx_m]>0.5 and 1==labelMat[0,idx_m]) or (Predict[0,idx_m]<0.5 and 0==labelMat[0,idx_m]):
            rightClassify+=1.0
    return Predict,rightClassify/float(m)

def plotBestFit(dataMatIn,Predict): 
    Xn= array(dataMatIn)
    N = shape(Xn)[0]

    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(N):
        #print Predict[0,i]
        if Predict[0,i]>= 0.5:
            xcord1.append(Xn[i,0]); ycord1.append(Xn[i,1])
        else:
            xcord2.append(Xn[i,0]); ycord2.append(Xn[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    return

if __name__ == '__main__':
    dataMatIn,classLabels=loadDataSet()
    W1,W2=BPTrainNetwork(dataMatIn,classLabels,2,4,1,500)
    print 'W1',W1
    print 'W2',W2
    Predict,count=NetworkPredict(dataMatIn,classLabels,W1,W2)
    print Predict
    print count
    plotBestFit(dataMatIn,Predict)
    pass