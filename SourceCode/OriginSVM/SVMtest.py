#coding=utf-8
'''
Created on 2013年11月11日

@author: Wangliaofan
'''
import SMOSVM
from numpy import *

if __name__ == '__main__':
    
    dataArr,dataLabel=SMOSVM.loadDataSet('testSet.txt')
    KernelType=['linear',2.0]
    C=0.9
    b,alphas,w=SMOSVM.smoPlatt(dataArr,dataLabel,C, 0.00001, 500,KernelType)
    print  b,w
    a=[]
    a.append(w[0,0])
    a.append(w[1,0]) 
    a.append(b[0,0])   
    print a
    SMOSVM.PlotSupportVectors(dataArr,dataLabel,a,alphas,C,KernelType)
    #SMOSVM.PlotSupportVectors(dataArr,dataLabel,a,alphas,C,KernelType)
    accuracy,DataNum=SMOSVM.CalculateAccuracy(a,dataArr,dataLabel)
    print accuracy,DataNum
    pass