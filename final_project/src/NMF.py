#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  pandas as pd
import  myparse as mp

from scipy import sparse
from sklearn.decomposition import PCA, NMF, TruncatedSVD


enroll_train = mp.readcsv("enrollment_train.csv")
enroll_test = mp.readcsv("enrollment_test.csv")

allthematrix = mp.readcsv("theMatrix.csv")

thematrix = allthematrix[1:,0:].astype(float)

print allthematrix[0,0:]

sm = sparse.csr_matrix(thematrix)
print type(sm)
#model = NMF(n_components=39)
model = TruncatedSVD(n_components=39)
newMatrix = model.fit_transform(np.transpose(sm))
#print model.reconstruction_err_ 
print (model.explained_variance_ratio_.sum())

newMatrix = np.vstack((allthematrix[0,0:],np.transpose(newMatrix)))



newM = pd.DataFrame(newMatrix[1:,0:].astype(float),columns=newMatrix[0,0:])
#print newM
feature_train = enroll_train[1:,0:]
feature_test = enroll_test[1:,0:]

#print feature_train

toStack = np.array([])
for i in range(0,39):
  toStack = np.append(toStack,['extra_course_feature_'+str(i)])
#print toStack

for i in range(1,np.shape(enroll_train)[0]):
  toStack = np.vstack((toStack,newM[enroll_train[i,2]].values))

print np.shape(toStack)
toSave = np.hstack((enroll_train,toStack))
np.savetxt("newfeat0108_train.csv", toSave, delimiter=",",fmt="%s")

  
toStack = np.array([])
for i in range(0,39):
  toStack = np.append(toStack,['extra_course_feature_'+str(i)])
#print toStack

for i in range(1,np.shape(enroll_test)[0]):
  toStack = np.vstack((toStack,newM[enroll_test[i,2]].values))
print np.shape(toStack)
toSave = np.hstack((enroll_test,toStack))
np.savetxt("newfeat0108_test.csv", toSave, delimiter=",",fmt="%s")
