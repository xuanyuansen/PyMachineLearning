#coding=utf-8
'''
Created on 2013年12月21日

@author: Wangliaofan
'''
from bs4 import BeautifulSoup
import urllib2
import numpy
import matplotlib.pyplot as plt

def lotteryplot(lotterymatrix):
    m,n=numpy.shape(lotterymatrix)
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    xcord3 = []; ycord3 = []
    for i in range(m):
        #xcord1.append(i); ycord1.append(truematrix[i,0])
        #xcord2.append(i); ycord2.append(truematrix[i,1])
        xcord3.append(i); ycord3.append(truematrix[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    ax.scatter(xcord3, ycord3, s=30, c='blue')
    plt.show()
    return

if __name__ == '__main__':
    sourceurl='http://baidu.lecai.com/lottery/draw/list/52/sjh'
    content = urllib2.urlopen(sourceurl).read()
    soup = BeautifulSoup(content)
    lotterylist=[]
    for spannames in soup.find_all("span"):
        #获得所有试机号和中奖号
        if spannames.get('class') and spannames.get('class')[0]=="result":
            contentofball=spannames.contents
            tempelement=[]
            tempelement.append(str(contentofball[1].contents[0]))
            tempelement.append(str(contentofball[3].contents[0]))
            tempelement.append(str(contentofball[5].contents[0]))
            lotterylist.append(tempelement)
    
    lotterylist=[map(int,i) for i in lotterylist]    
    #print lotterylist
    testlist=[]
    truelist=[]
    for i in range(len(lotterylist)):
        if 0==i%2:
            testlist.append(lotterylist[i])
        else:
            truelist.append(lotterylist[i])
            
    testlist.reverse()
    truelist.reverse()

    truematrix=numpy.mat(truelist)
    testmatrix=numpy.mat(testlist)
    
    m,n=numpy.shape(truematrix)
    tcount=numpy.mat(numpy.zeros((3,10)))
    for mi in range(m):
        for ni in range(n):
            a=truematrix[mi,ni]
            tcount[ni,a] += 1
    testcount=numpy.mat(numpy.zeros((3,10)))
    for mi in range(m):
        for ni in range(n):
            a=testmatrix[mi,ni]
            testcount[ni,a] += 1
    print tcount
    print testcount
    pass