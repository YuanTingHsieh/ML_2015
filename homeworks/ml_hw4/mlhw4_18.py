# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

lambo=0
lamb=1

train=np.loadtxt('hw4_train.dat')
test=np.loadtxt('hw4_test.dat')
dimen=np.shape(train)[1]-1
points= np.shape(train)[0]
testlen=np.shape(test)[0]

x_matrix=np.hstack((np.ones((np.shape(train)[0],1)),train[:,0:dimen]))
y=train[:,dimen]
x_test=np.hstack((np.ones((np.shape(test)[0],1)),test[:,0:dimen]))
y_test_real=test[:,dimen]

wlin=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix),x_matrix)+lamb*np.eye(np.shape(x_matrix)[1])),np.transpose(x_matrix)),y)
y_pred=np.sign(np.dot(x_matrix,wlin))
y_pred_test=np.sign(np.dot(x_test,wlin))
error=float(points-np.sum(np.equal(y,y_pred)))/points
eout=float(testlen-np.sum(np.equal(y_test_real,y_pred_test)))/testlen
print 'Ein is '+str(error)
print 'Eout is '+str(eout)
#plt.rcParams["figure.figsize"]=[20,10]
#plt.suptitle("Ein's frequency")
#plt.xlabel('Ein (rate)')
#plt.ylabel('frequency (counts)')
#his=plt.hist(all_ein,10)
#plt.show()
