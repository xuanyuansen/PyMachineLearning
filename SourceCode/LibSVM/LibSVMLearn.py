#coding=utf-8
'''
Created on 2013年9月20日
libsvm for 64 bit python from http://www.lfd.uci.edu/~gohlke/pythonlibs/#libsvm
@author: Wangliaofan
'''
import svmutil

if __name__ == '__main__':
    y, x = svmutil.svm_read_problem('heart_scale')
    m = svmutil.svm_train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = svmutil.svm_predict(y[200:], x[200:], m)
    pass