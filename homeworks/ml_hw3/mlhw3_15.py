# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

all_ein=np.array([])
all_eout=np.array([])
all_w=np.array([0,0,0,0,0,0])
all_w3=np.array([])
run_times=1000

for run in range(1,run_times+1):
  print '%4s' %run,
  print 'th run      \r',
  sys.stdout.flush()

  points=1000
  x0=np.repeat(np.array([1]),points)
  x1=np.random.uniform(-1.0,1.0,points)
  x2=np.random.uniform(-1.0,1.0,points)
  #old matrix
  #x_matrix=np.vstack((x0,x1,x2))
  x3=np.multiply(x1,x2)
  x4=np.power(x1,2)
  x5=np.power(x2,2)
  x_matrix=np.vstack((x0,x1,x2,x3,x4,x5))

  y_real=np.sign(np.power(x1,2)+np.power(x2,2)-0.6)
  thres=np.random.uniform(0.0,1.0,points)
  flip=(thres[:]<0.1)
  y=np.sign(np.power(x1,2)+np.power(x2,2)-0.6)
  y[flip]=(y[flip]*-1)
  wlin=np.dot(np.dot(np.linalg.inv(np.dot(x_matrix,np.transpose(x_matrix))),x_matrix),np.transpose(y))
  all_w=np.vstack((all_w,wlin))
  all_w3=np.append(all_w3,wlin[3])
  
  y_pred=np.sign(np.dot(np.transpose(wlin),x_matrix))
  #print y_pred
  error=float(points-np.sum(np.equal(y,y_pred)))/points
  all_ein=np.append(all_ein,error)
  
  newx0=np.repeat(np.array([1]),points)
  newx1=np.random.uniform(-1.0,1.0,points)
  newx2=np.random.uniform(-1.0,1.0,points)
  #old matrix
  #x_matrix=np.vstack((x0,x1,x2))
  newx3=np.multiply(newx1,newx2)
  newx4=np.power(newx1,2)
  newx5=np.power(newx2,2)
  newx_matrix=np.vstack((newx0,newx1,newx2,newx3,newx4,newx5))
  newy_real=np.sign(np.power(newx1,2)+np.power(newx2,2)-0.6)
  newthres=np.random.uniform(0.0,1.0,points)
  newflip=(newthres[:]<0.1)
  newy=np.sign(np.power(newx1,2)+np.power(newx2,2)-0.6)
  newy[newflip]=(newy[newflip]*-1)
  newy_pred=np.sign(np.dot(np.transpose(wlin),newx_matrix))
  newerror=float(points-np.sum(np.equal(newy,newy_pred)))/points
  all_eout=np.append(all_eout,newerror)
print 'W5 mean is '+str(np.mean(all_w[1:,5]))  
print 'W4 mean is '+str(np.mean(all_w[1:,4]))  
print 'W3 mean is '+str(np.mean(all_w[1:,3]))  
print 'W2 mean is '+str(np.mean(all_w[1:,2]))  
print 'W1 mean is '+str(np.mean(all_w[1:,1]))  
print 'W0 mean is '+str(np.mean(all_w[1:,0]))  
print 'Ein mean is '+str(np.mean(all_ein))  
print 'Eout mean is '+str(np.mean(all_eout))  
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Eout's frequency")
plt.xlabel('Eout (rate)')
plt.ylabel('frequency (counts)')
his=plt.hist(all_eout,10)
plt.show()
