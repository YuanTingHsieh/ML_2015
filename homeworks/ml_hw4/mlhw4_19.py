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
num_folds = 5
subset_size = points/num_folds

all_etrain=np.array([])
all_eout=np.array([])
all_eval=np.array([])

all_x_train=np.hstack((np.ones((np.shape(train)[0],1)),train[:,0:dimen]))

x_test = np.hstack((np.ones((np.shape(test)[0],1)),test[:,0:dimen]))
y_test = test[:,dimen]

for lamb_i in lamb:
  erreval=0
  for fold in range(1,num_folds+1,1):
    start = (fold-1)*subset_size
    end = fold*subset_size
    x_val = all_x_train[start:end,:]
    y_val = train[start:end,dimen]
    if (start!=0)and(end!=points):
      x_matrix = np.vstack((all_x_train[0:start ,: ],all_x_train[end:points,:]))
      y  = np.hstack((train[0:start ,dimen], train[end:points,dimen]))
    elif start==0:
      x_matrix = all_x_train[end:points,:]
      y  = train[end:points,dimen]
    else:
      x_matrix = all_x_train[0:start ,: ]
      y  = train[0:start ,dimen]
    wlin=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_matrix),x_matrix)+lamb_i*np.eye(np.shape(x_matrix)[1])),np.transpose(x_matrix)),y)
    y_pred_val=np.sign(np.dot(x_val,wlin))
    erreval=erreval+float(subset_size-np.sum(np.equal(y_val,y_pred_val)))/subset_size
  all_eval=np.append(all_eval,erreval/num_folds)
print 'Ecv is '+str(all_eval)
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Ecv vs lambda")
plt.ylabel('Ecv (rate)')
plt.xlabel('lambda (log10)')
his=plt.scatter(lambo,all_eval)
plt.show()
