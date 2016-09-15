#coding=utf-8
'''
Created on 2013年9月20日
milk for 64 bit python from http://www.lfd.uci.edu/~gohlke/pythonlibs/#milk
Package Documentation https://pypi.python.org/pypi/milk/
@author: Wangliaofan
'''

import numpy as np
import milk

if __name__ == '__main__':
    features = np.random.rand(100,10) # 2d array of features: 100 examples of 10 features each
    labels = np.zeros(100)
    features[50:] += .5
    labels[50:] = 1
    confusion_matrix, names = milk.nfoldcrossvalidation(features, labels)
    print 'Accuracy:', confusion_matrix.trace()/float(confusion_matrix.sum())
    print names
    pass