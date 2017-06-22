# -*- coding: utf-8 -*-
import numpy as np
import sys
gg=np.loadtxt('ntumlone_hw2_hw2_train.dat')
gg_test=np.loadtxt('ntumlone_hw2_hw2_test.dat')
y_test=np.transpose(gg_test[:,9])

all_error=np.array([])
all_s=np.array([])
all_theta=np.array([])
all_error_test=np.array([])
best_dim=-1
best_dim_error=float('inf')

for run in range(0,9):
  print run
  points=100
  x=np.transpose(gg[:,run])
  y=np.transpose(gg[:,9])
  y=y[x.argsort()]
  x=np.sort(x,axis=None)
  
  midd=x[:-1] + np.diff(x)/2
  midd=np.insert(midd,0,-11)
  midd=np.append(midd,11)
  #print 'Midd are '+str(midd)
  
  best_error=float('inf')
  best_i=-1126
  best_s=-1126
  for i in midd:
    #print 'Theta is'+str(i)
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
  all_error=np.append(all_error,best_error)
  all_s=np.append(all_s,best_s)
  all_theta=np.append(all_theta,best_i)
  if best_error < best_dim_error:
    best_dim_error=best_error
    best_dim=run

x_test=np.transpose(gg_test[:,best_dim])
pred_test=all_s[best_dim]*np.sign(x_test-all_theta[best_dim])
error_test=float(1000-np.sum(np.equal(y_test,pred_test)))/1000

#print 'Error are '+str(all_error)
print 'Best dim '+str(best_dim)
print 'Sign is '+str(all_s[best_dim])
print 'Theta is '+str(all_theta[best_dim])
print 'Train Error is '+str(all_error[best_dim])
print 'Test Error is '+str(error_test)
