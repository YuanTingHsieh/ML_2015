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
all_chichi_train_learnOrder = mp.readcsv("feature_learn_order_train_chichi.csv")
all_chichi_test_learnOrder = mp.readcsv("feature_learn_order_test_chichi.csv")

all_users = np.append(enroll_train[1:,1],enroll_test[1:,1])
all_courses = np.append(enroll_train[1:,2],enroll_test[1:,2])

#all_0108_train = mp.readcsv("newfeat0108_train.csv")
#all_0108_test = mp.readcsv("newfeat0108_test.csv")

#all_bycourse_train = mp.readcsv("bycoursefeat_train.csv")
#all_bycourse_test = mp.readcsv("bycoursefeat_test.csv")
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
chichi_train_learnOrder = all_chichi_train_learnOrder[1:,1:].astype(float)
#bycourse_train = all_bycourse_train[1:,1].astype(float)
#ps_train = np.vstack((ps_train,ps2_train))
#data_train = np.hstack((data_train,a0108_train))
#data_train = np.hstack((data_train,np.transpose(ps_train)))

data_test = sample_test_x[1:,1:].astype(float)
aug_test = aug_graph_test[1:,1:].astype(float)
feat_test = all_feat_test[0:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
azure2_test = all_azure2_test[1:,2:33].astype(float)
chichi_test = all_chichi_test[1:,1:].astype(float)
chichi_test_learnOrder = all_chichi_test_learnOrder[1:,1:].astype(float)
#bycourse_test = all_bycourse_test[1:,1].astype(float)
#a0108_test = all_0108_test[1:,3:].astype(float)

#ps_test = pseudo1_test[0:,1].astype(float)
#ps2_test = pseudo2_test[0:,1].astype(float)
data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,feat_train))
data_train = np.hstack((data_train,azure_train))
data_train = np.hstack((data_train,azure2_train))
data_train = np.hstack((data_train,chichi_train))
data_train = np.hstack((data_train,chichi_train_learnOrder))

label_train = truth_train[0:,1].astype(float)
#ps_test = np.vstack((ps_test,ps2_test))

data_test = np.hstack((data_test,aug_test))
data_test = np.hstack((data_test,feat_test))
data_test = np.hstack((data_test,azure_test))
data_test = np.hstack((data_test,azure2_test))
data_test = np.hstack((data_test,chichi_test))
data_test = np.hstack((data_test,chichi_test_learnOrder))
#data_test = np.hstack((data_test,a0108_test))
#data_test = np.hstack((data_test,np.transpose(ps_test)))

#pseudoing
#data_train = np.vstack((data_train,data_test))
#label_train = np.append(label_train,ps_test)
#train_weight = (float(1)/data_train[0:,2])
#test_weight = (float(1)/data_test[0:,2])

#Pre-Processing
preprocess = StandardScaler()
preprocess.fit(np.vstack((data_train,data_test)))
data_train = preprocess.transform(data_train)
data_test = preprocess.transform(data_test)

#Adding User ID and Course ID
fuck_train = np.vstack((train_users_id,train_courses_id))
#fuck_train = np.vstack((fuck_train,bycourse_train))
data_train = np.hstack((np.transpose(fuck_train),data_train))
print np.shape(data_train)

fuck_test = np.vstack((test_users_id,test_courses_id))
#fuck_test = np.vstack((fuck_test,bycourse_test))
data_test = np.hstack((np.transpose(fuck_test),data_test))

#Naming columns
ichen = np.array(['*']*np.shape(feat_train)[1]).astype('|S24')
thename = np.array(['users_id','courses_id'])
#thename = np.append(thename,all_bycourse_train[0,1])
thename = np.append(thename,sample_train_x[0,1:])
thename = np.append(thename,aug_graph_train[0,1:])
thename = np.append(thename,ichen)
thename = np.append(thename,all_azure_train[0,2:])
thename = np.append(thename,all_azure2_train[0,2:33])
thename = np.append(thename,all_chichi_train[0,1:])

print np.shape(thename)
print np.shape(data_train)

#preprocess.fit(np.vstack((data_train,data_test)))
#data_train = preprocess.transform(data_train)
#data_test = preprocess.transform(data_test)

#pca =PCA(n_components=220)
#data_train = pca.fit_transform(data_train)
#data_test = pca.transform(data_test)
'''
param = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7}
num_round = int(sys.argv[1])
#num_round = 350
plst = param.items()
print param
print 'num_round is'+str(num_round)
#skf = KFold(len(label_train), n_folds=5)
skf = StratifiedKFold(label_train, n_folds=5,shuffle=True)
#CV ing
corr=np.array([])
print ('running cross validation')
for train_index, test_index in skf:
    dtrain = xgb.DMatrix(data_train[train_index], label=label_train[train_index], weight=(float(1)/data_train[train_index][0:,5]))
    dtest = xgb.DMatrix(data_train[test_index],label=label_train[test_index], weight=(float(1)/data_train[test_index][0:,5]))
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    bst = xgb.train(param,dtrain,num_round,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    total = np.shape(labels)[0]
    preds[preds>0.5]=1
    preds[preds<=0.5]=0
    cor =  (total-sum(abs(preds-labels)))/total
    print 'Corr = %0.6f' % cor
    corr=np.append(corr,cor)
    #print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
print 'Mean Precision is %0.6f' %np.mean(corr) 
'''
dtrain = xgb.DMatrix(data_train,  label=label_train)
param = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7,'nthread':4}
num_round = 500

bst = xgb.train(param,dtrain,num_round)

dummy = np.zeros(np.shape(data_test)[0])
dtest = xgb.DMatrix(data_test,label= dummy)
pred_train = bst.predict(dtrain)
pred_train[pred_train>0.5]=1
pred_train[pred_train<=0.5]=0
cor =  (len(label_train)-sum(abs(pred_train-label_train)))/len(label_train)
print 'Corr = %0.6f' % cor
pred_test = bst.predict(dtest)
pred_test[pred_test>0.5]=1
pred_test[pred_test<=0.5]=0
print np.shape(pred_test)

f = open('XGB_weighted_500_0112_cla_train_chichi.csv','wb')
for i in range(0,len(sample_train_x)-1):
    f.write(str(sample_train_x[i+1,0])+','+str(int(pred_train[i]))+'\n')
f = open('XGB_weighted_500_0112_cla_test_chichi.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str(int(pred_test[i]))+'\n')

'''
print ('running cross validation, with customized loss function')
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'accu', float(sum(labels != (preds > 0.0))) / len(labels)

param = {'max_depth':2, 'eta':1, 'silent':1}
# train with customized objective
xgb.cv(param, dtrain, num_round, nfold = 5, seed = 0,
       obj = logregobj, feval=evalerror)
'''
