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
all_etrain=np.array([])
all_eout=np.array([])
all_eval=np.array([])

all_x_train=np.hstack((np.ones((np.shape(train)[0],1)),train[:,0:dimen]))
x_matrix=all_x_train[0:120,:]
y=train[0:120,dimen]
x_val = all_x_train[120:200,:]
y_val = train[120:200,dimen]
x_test=np.hstack((np.ones((np.shape(test)[0],1)),test[:,0:dimen]))
y_test_real=test[:,dimen]

for lamb_i in lamb:
  wlin=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix),x_matrix)+lamb_i*np.eye(np.shape(x_matrix)[1])),np.transpose(x_matrix)),y)
  y_pred=np.sign(np.dot(x_matrix,wlin))
  y_pred_val=np.sign(np.dot(x_val,wlin))
  y_pred_test=np.sign(np.dot(x_test,wlin))
  etrain=float(120-np.sum(np.equal(y,y_pred)))/120
  erreval=float(80-np.sum(np.equal(y_val,y_pred_val)))/80
  eout=float(testlen-np.sum(np.equal(y_test_real,y_pred_test)))/testlen
  all_etrain=np.append(all_etrain,etrain)
  all_eval=np.append(all_eval,erreval)
  all_eout=np.append(all_eout,eout)
print 'Etrain is '+str(all_etrain)
print 'Eval is '+str(all_eval)
print 'Eout is '+str(all_eout)
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Eval vs lambda")
plt.ylabel('Eval (rate)')
plt.xlabel('lambda (log10)')
his=plt.scatter(lambo,all_eval)
plt.show()
