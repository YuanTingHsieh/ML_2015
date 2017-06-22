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

all_svd_train = mp.readcsv("newfeatwithsvd0111_train.csv")
all_svd_test = mp.readcsv("newfeatwithsvd0111_test.csv")

all_bycourse_train = mp.readcsv("bycoursefeat_train.csv")
all_bycourse_test = mp.readcsv("bycoursefeat_test.csv")

all_users = np.append(enroll_train[1:,1],enroll_test[1:,1])
all_courses = np.append(enroll_train[1:,2],enroll_test[1:,2])


#Start Encode Courses and Users
llle = LabelEncoder()

users_id = llle.fit_transform(all_users).astype(float)
courses_id = llle.fit_transform(all_courses).astype(float)

train_users_id = users_id[0:len(sample_train_x)-1]
test_users_id = users_id[(len(sample_train_x)-1):]
train_courses_id = courses_id[0:len(sample_train_x)-1]
test_courses_id = courses_id[(len(sample_train_x)-1):]


#Segmenting data to use
data_train = sample_train_x[1:,1:].astype(float)
aug_train = aug_graph_train[1:,1:].astype(float)
feat_train = all_feat_train[0:,1:].astype(float)
azure_train = all_azure_train[1:,2:].astype(float)
azure2_train = all_azure2_train[1:,2:33].astype(float)
chichi_train = all_chichi_train[1:,1:].astype(float)
bycourse_train = all_bycourse_train[1:,1].astype(float)
svd_train = all_svd_train[0:,0:].astype(float)

data_test = sample_test_x[1:,1:].astype(float)
aug_test = aug_graph_test[1:,1:].astype(float)
feat_test = all_feat_test[0:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
azure2_test = all_azure2_test[1:,2:33].astype(float)
chichi_test = all_chichi_test[1:,1:].astype(float)
bycourse_test = all_bycourse_test[1:,1].astype(float)
svd_test = all_svd_test[0:,0:].astype(float)

data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,feat_train))
data_train = np.hstack((data_train,azure_train))
data_train = np.hstack((data_train,azure2_train))
data_train = np.hstack((data_train,chichi_train))
data_train = np.hstack((data_train,svd_train))

label_train = truth_train[0:,1].astype(float)

data_test = np.hstack((data_test,aug_test))
data_test = np.hstack((data_test,feat_test))
data_test = np.hstack((data_test,azure_test))
data_test = np.hstack((data_test,azure2_test))
data_test = np.hstack((data_test,chichi_test))
data_test = np.hstack((data_test,svd_test))

#Preprocessing
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

print np.shape(thename)
print np.shape(data_train)

#Below for training
'''
param = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7,'eval_metric':'auc','nthread':5}
num_round = int(sys.argv[1])
plst = param.items()
print param
print 'num_round is'+str(num_round)

skf = StratifiedKFold(label_train, n_folds=5,shuffle=True)
#CV ing
corr=np.array([])
print ('running cross validation')
for train_index, test_index in skf:
    dtrain = xgb.DMatrix(data_train[train_index], label=label_train[train_index])
    dtest = xgb.DMatrix(data_train[test_index],label=label_train[test_index])
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    bst = xgb.train(param,dtrain,num_round,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    total = np.shape(labels)[0]
    #preds[preds>0.5]=1
    #preds[preds<=0.5]=0
    #cor =  (total-sum(abs(preds-labels)))/total
    cor = roc_auc_score(labels,preds)
    print 'AUC = %0.6f' % cor
    corr=np.append(corr,cor)
    #print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
print 'Mean AUC is %0.6f' %np.mean(corr) 

'''
dtrain = xgb.DMatrix(data_train,  label=label_train)
param = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'rank:pairwise','subsample':0.8,'colsample_bytree':0.7,'eval_metric':'map','nthread':4}
num_round = 300

bst = xgb.train(param,dtrain,num_round)


dummy = np.zeros(np.shape(data_test)[0])
dtest = xgb.DMatrix(data_test,label= dummy)
pred_train = bst.predict(dtrain)
#pred_train[pred_train>0.5]=1
#pred_train[pred_train<=0.5]=0
pred_test = bst.predict(dtest)
#pred_test[pred_test>0.5]=1
#pred_test[pred_test<=0.5]=0
print "Train AUC is "+str(roc_auc_score(label_train,pred_train))


'''
#oop=xgb.plot_importance(bst)
importance=bst.get_fscore()
#plt.savefig('ggg.png')
tuples = [(k, importance[k]) for k in importance]
tuples = sorted(tuples, key=lambda x: x[1])
opopop = np.zeros(np.shape(data_train)[1]).astype(bool)
for i in tuples:
  wtff = int(filter(str.isdigit, i[0]))
  opopop[wtff]=True
gg = range(0,np.shape(data_train)[1])
for i in range(0,len(opopop)):
  if opopop[i] ==False:
    print 'dim '+str(gg[i])+'is not in tuple'
    print 'name is '+thename[i]
print 'Total dim used' + str(len(tuples))
#print tuples

new_data_train = data_train[0:,opopop]
new_data_test = data_test[0:,opopop]

newdtrain = xgb.DMatrix(new_data_train, label = label_train)
newdtest = xgb.DMatrix(new_data_test,label = dummy)

param2 = {'max_depth':5, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':0.7,'eval_metric':'auc'}
num_round2 = 450

bst = xgb.train(param2,newdtrain,num_round2)

pred_train = bst.predict(newdtrain)
pred_test = bst.predict(newdtest)

print "New AUC is "+str(roc_auc_score(label_train,pred_train))
'''

f = open('XGB_5_005_08_07_300_all_0111_mapout_train.csv','wb')
for i in range(0,len(pred_train)):
    f.write(str(sample_train_x[i+1,0])+','+str((pred_train[i]))+'\n')
f = open('XGB_5_005_08_07_300_all_0111_mapout_test.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str((pred_test[i]))+'\n')

