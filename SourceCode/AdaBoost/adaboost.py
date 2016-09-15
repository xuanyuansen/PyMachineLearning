#coding=utf-8
'''
Created on 2014年4月13日

@author: Wangliaofan
'''
import random
import numpy
def alpha_errorrate(errornum,totalnum):
    error=float(errornum)/float(totalnum)
    return 0.5*numpy.log((1-error)/(error))
#从n个数中随机选出m个数
def genknuth(m,n):
    for i in range(0,n):
        if ( random.randint(0,n)%(n-i) )< m:
            print i
            m=m-1
    return
if __name__ == '__main__':
    genknuth(20,100)
    al=alpha_errorrate(10,100)
    print al
    pass