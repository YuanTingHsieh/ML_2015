# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

all_error=np.array([])
eout=np.array([])
for run in range(1,5001):
  print '%4s' %run,
  print 'th run      \r',
  sys.stdout.flush()

  points=20
  x=np.random.uniform(-1.0,1.0,points)
  x=np.sort(x,axis=None)
  
  z=range(0,points)
  
  y_real = np.sign(x)
  thres=np.random.uniform(0.0,1.0,points)
  flip=(thres[:]<0.2)
  y=np.sign(x)
  y[flip]=(y[flip]*-1)
  #print 'After flip Y is '+str(y)
  
  midd=x[:-1] + np.diff(x)/2
  midd=np.insert(midd,0,-1.0)
  midd=np.append(midd,1.0)
  #print 'Midd are '+str(midd)
  
  best_error=float('inf')
  best_i=-1126
  best_s=-1126
  for i in midd:
    s=1.0
    pred=s*np.sign(x-i)
    error=float(points-np.sum(np.equal(y,pred)))/points
    if error<best_error:
      best_i=i
      best_s=s
      best_error=error
    s=-1.0 
    pred=s*np.sign(x-i)
    error=float(points-np.sum(np.equal(y,pred)))/points
    if error<best_error:
      best_i=i
      best_s=s
      best_error=error
  #print best_i,best_s,best_error
  eout=np.append(eout,0.5+0.3*best_s*(np.absolute(best_i)-1))
  all_error=np.append(all_error,best_error)
plt.rcParams["figure.figsize"]=[20,10]
plt.suptitle("Ein's frequency")
plt.xlabel('Ein (rate)')
plt.ylabel('frequency (counts)')
plt.xticks(np.arange(0,0.5,0.05))
#plt.legend()
#print str(all_error)
bins=-0.025+0.05*np.array(range(0,11))
his=plt.hist(all_error,bins)
#print his
print 'Ein mean is '+str(np.mean(all_error))
print 'Eout mean is '+str(np.mean(eout))
plt.show()
plt.suptitle("Eout's frequency")
plt.xlabel('Eout (rate)')
plt.ylabel('frequency (counts)')
his=plt.hist(eout,bins=10)
plt.show()
