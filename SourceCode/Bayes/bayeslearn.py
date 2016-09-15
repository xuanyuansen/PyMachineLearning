#coding=utf-8
'''
Created on 2013年9月20日

@author: Wangliaofan
'''
import bayes
import feedparser
from time import *

if __name__== '__main__':
    listOPosts,listClasses = bayes.loadDataSet()
    print listOPosts,listClasses
    myVocabList = bayes.createVocabList(listOPosts)
    print myVocabList
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    print trainMat
    p0V,p1V,pAb=bayes.trainNB0(trainMat, listClasses)
    print p0V
    print p1V
    print pAb
    #ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    #sleep(5)
    #print ny['entries']
    bayes.spamTest()
    pass