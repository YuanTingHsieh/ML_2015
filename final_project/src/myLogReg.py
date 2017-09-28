#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  myparse as mp
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA

# read csv include first line
enroll_train = mp.readcsv("enrollment_train.csv")
truth_train = mp.readcsv("truth_train.csv")
sample_train_x = mp.readcsv("sample_train_x.csv")
sample_test_x = mp.readcsv("sample_test_x.csv")
aug_graph_train = mp.readcsv("augmentGraph_train.csv")
aug_graph_test =mp.readcsv("augmentGraph_test.csv")
all_azure_train = mp.readcsv("azure_train.csv")
all_azure_test = mp.readcsv("azure_test.csv")

all_feat_train = mp.readcsv("feat.csv")

aug_train = aug_graph_train[1:,1:].astype(float)
data_train = sample_train_x[1:,1:].astype(float)
#feat_train = all_feat_train[0:,1:].astype(float)
azure_train = all_azure_train[1:,2:].astype(float)

data_train = np.hstack((data_train,aug_train))
#data_train = np.hstack((data_train,aug_feat))
data_train = np.hstack((data_train,azure_train))
print np.shape(data_train)
label_train = truth_train[0:,1].astype(float)

aug_test = aug_graph_test[1:,1:].astype(float)
data_test = sample_test_x[1:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
data_test = np.hstack((data_test,aug_test))
data_test = np.hstack((data_test,azure_test))

#Pre-Processing
preprocess = StandardScaler()
#preprocess = RobustScaler()

data_train = preprocess.fit_transform(data_train)
data_test = preprocess.fit_transform(data_test)

#no PCA -> (C=1e-5,tol=0.1,corr=0.852313)
#PCA 15(my+sample) -> (C=1e-5,tol=0.1,corr=0.852417)


#doing PCA
#pca =PCA(n_components=45) #(C=1e-5,tol=0.1,0.852417)
#data_train = pca.fit_transform(data_train)
#data_test = pca.transform(data_test)


parameters = {'C':np.power(10.0,range(-5,6,1)).tolist(),'tol':np.power(10.0,range(-6,1,1)).tolist(),'dual':[False],'n_jobs':[4],'solver':['sag'],'max_iter':[500]}
print "Tuning parameters"
svc = LogisticRegression()
clf = grid_search.GridSearchCV(svc, parameters,cv=5)
clf.fit(data_train, label_train) 
print "Best parameters set found on development set:"
print ""
print(clf.best_params_)
print ""
for params, mean_score, scores in clf.grid_scores_:
    print "%0.6f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)
'''

clf = svm.LinearSVC(C=1e-4,tol=0.1,dual=False)
scores = cross_validation.cross_val_score(clf,data_train,label_train,cv=5)
print("Accuracy: %0.6f (+/- %0.03f)" % (scores.mean(), scores.std() * 2))
'''
'''
clf.fit(data_train,label_train)
pred_test = clf.predict(data_test)
print np.shape(pred_test)

f = open('sklinearClassifywithGraphPCA15c0000001e01.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str(pred_test[i].astype(int))+'\n')

'''
