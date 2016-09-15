#coding=utf-8
'''
Created on 2013年11月30日

@author: Wangliaofan
'''
from numpy import *


def loadDataSet():
    #数据来源于Data Mining Chapter 6， page 250， Table 6.1
    data=[[1, 2, 5], [2, 4], [ 2, 3], [1,2, 4],[1,3],[2,3],[1,3],[1,2,3,5],[1,2,3],\
    #第二行是自己增加的      
          [1,2,3,4,5,6],[1,2,3],[2,4,6],[3,4,6],[1,2,5],[1,5,6],[1,2,3,5,6]]
    maxlen=0
    for subdata in data:
        currentlen=len(subdata)
        if currentlen>maxlen:
            maxlen=currentlen
    return data,maxlen

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    print C1
    C1.sort()
    print C1
    return map(frozenset, C1)#use frozen set so we can use it as a key in a dict

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    #numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        #support = ssCnt[key]/numItems
        support = ssCnt[key]
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    retList.sort()
    return retList, supportData

def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
                
    return retList

def apriori(dataSet, maxlen,minSupport = 0.5 ):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    #while (len(L[k-2]) > 0):
    while (maxlen >= k):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        if len(Lk)>=1:
            Lk.sort()
            L.append(Lk)
        else:
            return L, supportData
        k += 1
    return L, supportData

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = float(supportData[freqSet])/float(supportData[freqSet-conseq]) #calc confidence
        print freqSet,freqSet-conseq
        print conf
        #print freqSet-conseq,supportData[freqSet-conseq]
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    print 'H',H
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        print i
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList    
   
if __name__ == '__main__':

    data,maxlen=loadDataSet()
    print data
    C1= createC1(data)
    print C1
    D=map(set,data)
    print D
    minSupport=2
    L1,supportData0=scanD(D, C1, minSupport)
    print L1,supportData0
    #L2=aprioriGen(L1, 2)
    #print L2
    L, supportData= apriori(data,maxlen, minSupport )
    print L, supportData
    print len(L)
    rules=generateRules(L, supportData)
    for item in rules:
        print item
    pass