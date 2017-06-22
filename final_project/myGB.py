#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  myparse as mp
from sklearn import cross_validation, grid_search
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



# read csv include first line
enroll_train = mp.readcsv("enrollment_train.csv")
truth_train = mp.readcsv("truth_train.csv")
sample_train_x = mp.readcsv("sample_train_x.csv")
sample_test_x = mp.readcsv("sample_test_x.csv")
aug_graph_train = mp.readcsv("augmentGraph_train.csv")
aug_graph_test =mp.readcsv("augmentGraph_test.csv")

aug_feat_train = mp.readcsv("feat.csv")

aug_train = aug_graph_train[1:,1:].astype(float)
data_train = sample_train_x[1:,1:].astype(float)
aug_feat = aug_feat_train[0:,1:].astype(float)

data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,aug_feat))
print np.shape(data_train)
label_train = truth_train[0:,1].astype(float)

aug_test = aug_graph_test[1:,1:].astype(float)
data_test = sample_test_x[1:,1:].astype(float)
data_test = np.hstack((data_test,aug_test))

#Pre-Processing
preprocess = StandardScaler()
#preprocess = RobustScaler()

data_train = preprocess.fit_transform(data_train)
data_test = preprocess.fit_transform(data_test)

#no PCA -> (C=1e-5,tol=0.1,corr=0.852313)
#PCA 15(my+sample) -> (C=1e-5,tol=0.1,corr=0.852417)


#doing PCA
#pca =PCA(n_components=40) #(C=1e-5,tol=0.1,0.852417)
#data_train = pca.fit_transform(data_train)
#data_test = pca.transform(data_test)

parameters = {'learning_rate':[0.01,0.02],'n_estimators':range(1000,2200,200),'max_leaf_nodes':[100,200],'min_samples_leaf':[4,5]}
svc = GradientBoostingClassifier()
print "Tuning parameters"
clf = grid_search.GridSearchCV(svc, parameters,cv=5)
clf.fit(data_train, label_train) 
print "Best parameters set found on development set:"
print ""
print(clf.best_params_)
print ""
for params, mean_score, scores in clf.grid_scores_:
    print "%0.6f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)


