#coding=utf-8
'''
Created on 2013年12月29日

@author: Wangliaofan
'''

def LoadLibData(targetfilename="heart_scale"):
    dataMat = []; labelMat = []
    tarfile = open(targetfilename,'r')
    MaxDimension=0
    for line in tarfile.readlines():
        lineArray = line.strip().split()
        labelMat.append(float(lineArray[0]))
        RowDataTemp=[]
        for element_iter in range(1,len(lineArray)):
            MaxDimension=max(MaxDimension,len(lineArray)-1)
            TempStringArr=lineArray[element_iter].split(':')
            RowDataTemp.append(float(TempStringArr[1]))
        dataMat.append(RowDataTemp)
    #print MaxDemension
    for element_iter in range(0,len(dataMat)):
        for sub_iter in range(MaxDimension-len(dataMat[element_iter])):
            dataMat[element_iter].append(0.0)
    return dataMat,labelMat,MaxDimension

if __name__ == '__main__':
    #print max(2,1)
    dataMat,labelMat=LoadLibData()
    print dataMat,labelMat
    pass