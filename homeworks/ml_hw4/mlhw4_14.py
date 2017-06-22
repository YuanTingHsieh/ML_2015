# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys


lambo=np.arange(-10,3,1)
lamb=np.power(float(10),lambo)

train=np.loadtxt('hw4_train.dat')
test=np.loadtxt('hw4_test.dat')
dimen=np.shape(train)[1]-1
points= np.shape(train)[0]
testlen=np.shape(test)[0]
all_ein=np.array([])
all_eout=np.array([])

x_matrix=np.hstack((np.ones((np.shape(train)[0],1)),train[:,0:dimen]))
y=train[:,dimen]
x_test=np.hstack((np.ones((np.shape(test)[0],1)),test[:,0:dimen]))
y_test_real=test[:,dimen]

for lamb_i in lamb:
  wlin=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix),x_matrix)+lamb_i*np.eye(np.shape(x_matrix)[1])),np.transpose(x_matrix)),y)
  y_pred=np.sign(np.dot(x_matrix,wlin))
  y_pred_test=np.sign(np.dot(x_test,wlin))
  error=float(points-np.sum(np.equal(y,y_pred)))/points
  eout=float(testlen-np.sum(np.equal(y_test_real,y_pred_test)))/testlen
  all_ein=np.append(all_ein,error)
  all_eout=np.append(all_eout,eout)
print 'Ein is '+str(all_ein)
print 'Eout is '+str(all_eout)
print 'Ein mean is '+str(np.mean(all_ein))
print 'Eout mean is '+str(np.mean(all_eout))
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Ein vs lambda")
plt.ylabel('Ein (rate)')
plt.xlabel('lambda (log10)')
his=plt.scatter(lambo,all_ein)
plt.show()
