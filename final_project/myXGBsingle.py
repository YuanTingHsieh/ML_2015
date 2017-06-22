#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import myparse as mp
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# read csv include first line
enroll_train = mp.readcsv("enrollment_train.csv")
enroll_test = mp.readcsv("enrollment_test.csv")

truth_train = mp.readcsv("truth_train.csv")
sample_train_x = mp.readcsv("sample_train_x.csv")
sample_test_x = mp.readcsv("sample_test_x.csv")
aug_graph_train = mp.readcsv("augmentGraph_train.csv")
aug_graph_test =mp.readcsv("augmentGraph_test.csv")

all_feat_train = mp.readcsv("feat_train.csv")
all_feat_test = mp.readcsv("feat_test.csv")

all_azure_train = mp.readcsv("azure_train.csv")
all_azure_test = mp.readcsv("azure_test.csv")

all_azure2_train = mp.readcsv("azure2_train.csv")
all_azure2_test = mp.readcsv("azure2_test.csv")

all_chichi_train = mp.readcsv("feature_train_chichi.csv")
all_chichi_test = mp.readcsv("feature_test_chichi.csv")

all_users = np.append(enroll_train[1:,1],enroll_test[1:,1])
all_courses = np.append(enroll_train[1:,2],enroll_test[1:,2])

#all_0108_train = mp.readcsv("newfeat0108_train.csv")
#all_0108_test = mp.readcsv("newfeat0108_test.csv")

all_bycourse_train = mp.readcsv("bycoursefeat_train.csv")
all_bycourse_test = mp.readcsv("bycoursefeat_test.csv")
#Start Encode Courses and Users
llle = LabelEncoder()

users_id = llle.fit_transform(all_users).astype(float)
courses_id = llle.fit_transform(all_courses).astype(float)

train_users_id = users_id[0:len(sample_train_x)-1]
test_users_id = users_id[(len(sample_train_x)-1):]
train_courses_id = courses_id[0:len(sample_train_x)-1]
test_courses_id = courses_id[(len(sample_train_x)-1):]
#pseudo1_train = mp.readcsv("XGB_4_005_08_06_300_all_reg_train.csv")
#pseudo1_test = mp.readcsv("XGB_4_005_08_06_300_all_cla_test.csv")


#pseudo2_train = mp.readcsv("XGB_5_005_08_06_300_all0107_reg_train.csv")
#pseudo2_test = mp.readcsv("XGB_5_005_08_06_300_all0107_reg_test.csv")


#Segmenting data to use
data_train = sample_train_x[1:,1:].astype(float)
aug_train = aug_graph_train[1:,1:].astype(float)
feat_train = all_feat_train[0:,1:].astype(float)
azure_train = all_azure_train[1:,2:].astype(float)
azure2_train = all_azure2_train[1:,2:33].astype(float)
chichi_train = all_chichi_train[1:,1:].astype(float)
bycourse_train = all_bycourse_train[1:,1].astype(float)
#ps_train = np.vstack((ps_train,ps2_train))
#data_train = np.hstack((data_train,a0108_train))
#data_train = np.hstack((data_train,np.transpose(ps_train)))

data_test = sample_test_x[1:,1:].astype(float)
aug_test = aug_graph_test[1:,1:].astype(float)
feat_test = all_feat_test[0:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
azure2_test = all_azure2_test[1:,2:33].astype(float)
chichi_test = all_chichi_test[1:,1:].astype(float)
bycourse_test = all_bycourse_test[1:,1].astype(float)
#a0108_test = all_0108_test[1:,3:].astype(float)

#ps_test = pseudo1_test[0:,1].astype(float)
#ps2_test = pseudo2_test[0:,1].astype(float)
data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,feat_train))
data_train = np.hstack((data_train,azure_train))
data_train = np.hstack((data_train,azure2_train))
data_train = np.hstack((data_train,chichi_train))

label_train = truth_train[0:,1].astype(float)
#ps_test = np.vstack((ps_test,ps2_test))

data_test = np.hstack((data_test,aug_test))
data_test = np.hstack((data_test,feat_test))
data_test = np.hstack((data_test,azure_test))
data_test = np.hstack((data_test,azure2_test))
data_test = np.hstack((data_test,chichi_test))
#data_test = np.hstack((data_test,a0108_test))
#data_test = np.hstack((data_test,np.transpose(ps_test)))

#pseudoing
#data_train = np.vstack((data_train,data_test))
#label_train = np.append(label_train,ps_test)

#Single and Mutiple 
index_train_single = data_train[0:,2]==1
index_train_multiple = data_train[0:,2]!=1
index_test_single =data_test[0:,2]==1
index_test_multiple = data_test[0:,2]!=1

print data_train[0:5,2]

#Pre-Processing
preprocess = StandardScaler()
preprocess.fit(np.vstack((data_train,data_test)))
data_train = preprocess.transform(data_train)
data_test = preprocess.transform(data_test)

#Adding User ID and Course ID
fuck_train = np.vstack((train_users_id,train_courses_id))
fuck_train = np.vstack((fuck_train,bycourse_train))
data_train = np.hstack((np.transpose(fuck_train),data_train))
print np.shape(data_train)

fuck_test = np.vstack((test_users_id,test_courses_id))
fuck_test = np.vstack((fuck_test,bycourse_test))
data_test = np.hstack((np.transpose(fuck_test),data_test))

#Naming columns
ichen = np.array(['*']*np.shape(feat_train)[1]).astype('|S24')
thename = np.array(['users_id','courses_id'])
thename = np.append(thename,all_bycourse_train[0,1])
thename = np.append(thename,sample_train_x[0,1:])
thename = np.append(thename,aug_graph_train[0,1:])
thename = np.append(thename,ichen)
thename = np.append(thename,all_azure_train[0,2:])
thename = np.append(thename,all_azure2_train[0,2:33])
thename = np.append(thename,all_chichi_train[0,1:])

print thename[5]
print np.shape(data_train)

data_train_single = data_train[index_train_single,0:]
data_train_multiple = data_train[index_train_multiple,0:]

label_train_single = label_train[index_train_single]
label_train_multiple  = label_train[index_train_multiple]

print "Total Train Single user are "+str(len(data_train_single))
print "Total Train Multiple user are "+str(len(data_train_multiple))

#preprocess.fit(np.vstack((data_train,data_test)))
#data_train = preprocess.transform(data_train)
#data_test = preprocess.transform(data_test)

#pca =PCA(n_components=220)
#data_train = pca.fit_transform(data_train)
#data_test = pca.transform(data_test)
'''
param1 = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':1.0,'colsample_bytree':0.7,'nthread':4,'seed':611}
num_round = 50
num_round1 = int(sys.argv[1])
#num_round = 350
print param1
print 'num_round is'+str(num_round)
skf1 = StratifiedKFold(label_train_single, n_folds=5,shuffle=True)
#CV ing
corr=np.array([])
print ('running cross validation')
for train_index, test_index in skf1:
    dtrain = xgb.DMatrix(data_train_single[train_index], label=label_train_single[train_index])
    dtest = xgb.DMatrix(data_train_single[test_index],label=label_train_single[test_index])
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    bst = xgb.train(param1,dtrain,num_round,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    total = np.shape(labels)[0]
    preds[preds>0.5]=1
    preds[preds<=0.5]=0
    cor =  float(sum(preds==labels))/total
    print 'Single Corr = %0.6f' % cor
    corr=np.append(corr,cor)

print 'Mean Single Precision is %0.6f' %np.mean(corr)
param2 = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.85,'colsample_bytree':0.7,'nthread':4}
skf2 = StratifiedKFold(label_train_multiple,n_folds=5,shuffle=True)
#CV ing
corr2=np.array([])
print ('running cross validation')
for train_index, test_index in skf2:
    dtrain = xgb.DMatrix(data_train_multiple[train_index], label=label_train_multiple[train_index])
    dtest = xgb.DMatrix(data_train_multiple[test_index],label=label_train_multiple[test_index])
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    bst = xgb.train(param2,dtrain,num_round1,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    total = np.shape(labels)[0]
    preds[preds>0.5]=1
    preds[preds<=0.5]=0
    cor =  float(sum(preds==labels))/total
    print 'Multi Corr = %0.6f' % cor
    corr2=np.append(corr2,cor)

print 'Mean Single Precision is %0.6f' %np.mean(corr)
print 'Mean Multiple Precision is %0.6f' %np.mean(corr2)

'''

dtrain_single = xgb.DMatrix(data_train_single,  label=label_train_single)
dtrain_multiple = xgb.DMatrix(data_train_multiple,label=label_train_multiple)

dtest_single = xgb.DMatrix(data_test[index_test_single,0:], label=np.zeros(np.sum(index_test_single)))
dtest_multiple = xgb.DMatrix(data_test[index_test_multiple,0:], label=np.zeros(np.sum(index_test_multiple)))

param = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7,'nthread':6}
param1 = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7,'nthread':6}
num_round = 400
num_round1 = 400

bst_single = xgb.train(param,dtrain_single,num_round)
bst_multiple = xgb.train(param1,dtrain_multiple,num_round1)

pred_train_single = bst_single.predict(dtrain_single)
#pred_train_single[pred_train_single>0.5]=1
#pred_train_single[pred_train_single<=0.5]=0

pred_train_multiple = bst_multiple.predict(dtrain_multiple)
#pred_train_multiple[pred_train_multiple>0.5]=1
#pred_train_multiple[pred_train_multiple<=0.5]=0

pred_train = np.zeros(len(pred_train_multiple)+len(pred_train_single))
pred_train[index_train_single]=pred_train_single
pred_train[index_train_multiple]=pred_train_multiple


print 'Train single corr = %0.6f' % (float(sum(pred_train_single==label_train_single))/len(label_train_single))
print 'Train Multiple corr = %0.6f' % (float(sum(pred_train_multiple==label_train_multiple))/len(label_train_multiple))
print 'Train Corr = %0.6f' % (float(sum(pred_train==label_train))/len(label_train))

pred_test_single = bst_single.predict(dtest_single)
pred_test_single[pred_test_single>0.5]=1
pred_test_single[pred_test_single<=0.5]=0

pred_test_multiple = bst_multiple.predict(dtest_multiple)
pred_test_multiple[pred_test_multiple>0.5]=1
pred_test_multiple[pred_test_multiple<=0.5]=0

pred_test = np.zeros(len(pred_test_multiple)+len(pred_test_single))
pred_test[index_test_single]=pred_test_single
pred_test[index_test_multiple]=pred_test_multiple

f = open('XGB_5_005_08_09_400_all_0111_cla_single_train.csv','wb')
for i in range(0,len(sample_train_x)-1):
    f.write(str(sample_train_x[i+1,0])+','+str((pred_train[i]))+'\n')
f = open('XGB_5_005_08_09_400_all_0111_cla_single_test.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str((pred_test[i]))+'\n')

