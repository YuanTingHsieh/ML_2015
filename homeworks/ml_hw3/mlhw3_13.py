# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

all_ein=np.array([])
run_times=1000

for run in range(1,run_times+1):
  print '%4s' %run,
  print 'th run      \r',
  sys.stdout.flush()

  points=1000
  x0=np.repeat(np.array([1]),points)
  #print x0
  x1=np.random.uniform(-1.0,1.0,points)
  #print x1
  x2=np.random.uniform(-1.0,1.0,points)
  #print x2
  #x=np.sort(x,axis=None)
  x_matrix=np.vstack((x0,x1,x2))
  #print x_matrix

  y_real=np.sign(np.power(x1,2)+np.power(x2,2)-0.6)
  #print y_real  
  thres=np.random.uniform(0.0,1.0,points)
  flip=(thres[:]<0.1)
  #print flip
  y=np.sign(np.power(x1,2)+np.power(x2,2)-0.6)
  y[flip]=(y[flip]*-1)
  #print y_real
  #print str(y)
  wlin=np.dot(np.dot(np.linalg.inv(np.dot(x_matrix,np.transpose(x_matrix))),x_matrix),np.transpose(y))
  #print wlin
  y_pred=np.sign(np.dot(np.transpose(wlin),x_matrix))
  #print y_pred
  error=float(points-np.sum(np.equal(y,y_pred)))/points
  all_ein=np.append(all_ein,error)
print 'Ein mean is '+str(np.mean(all_ein))  
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Ein's frequency")
plt.xlabel('Ein (rate)')
plt.ylabel('frequency (counts)')
his=plt.hist(all_ein,10)
plt.show()
