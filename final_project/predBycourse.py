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

#Start Encode Courses and Users
llle = LabelEncoder()

users_id = llle.fit_transform(all_users).astype(float)
courses_id = llle.fit_transform(all_courses).astype(float)

train_users_id = users_id[0:len(sample_train_x)-1]
test_users_id = users_id[(len(sample_train_x)-1):]
train_courses_id = courses_id[0:len(sample_train_x)-1]
test_courses_id = courses_id[(len(sample_train_x)-1):]
#all_0108_train = mp.readcsv("newfeat0108_train.csv")
#all_0108_test = mp.readcsv("newfeat0108_test.csv")
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
#ps_train = np.vstack((ps_train,ps2_train))
#data_train = np.hstack((data_train,a0108_train))
#data_train = np.hstack((data_train,np.transpose(ps_train)))

data_test = sample_test_x[1:,1:].astype(float)
aug_test = aug_graph_test[1:,1:].astype(float)
feat_test = all_feat_test[0:,1:].astype(float)
azure_test = all_azure_test[1:,2:].astype(float)
azure2_test = all_azure2_test[1:,2:33].astype(float)
chichi_test = all_chichi_test[1:,1:].astype(float)
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

#Pre-Processing
preprocess = StandardScaler()
preprocess.fit(np.vstack((data_train,data_test)))
data_train = preprocess.transform(data_train)
data_test = preprocess.transform(data_test)

#Adding User ID and Course ID
fuck_train = np.vstack((train_users_id,train_courses_id))
data_train = np.hstack((np.transpose(fuck_train),data_train))

fuck_test = np.vstack((test_users_id,test_courses_id))
data_test = np.hstack((np.transpose(fuck_test),data_test))

#Naming columns
ichen = np.array(['*']*np.shape(feat_train)[1]).astype('|S24')
thename = np.array(['users_id','courses_id'])
thename = np.append(thename,sample_train_x[0,1:])
thename = np.append(thename,aug_graph_train[0,1:])
thename = np.append(thename,ichen)
thename = np.append(thename,all_azure_train[0,2:])
thename = np.append(thename,all_azure2_train[0,2:33])
thename = np.append(thename,all_chichi_train[0,1:])

print np.shape(data_train)

#tosave = np.hstack((enrollment_train[1:,0],new_bycourse))

param = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':1.0,'colsample_bytree':0.7,'nthread':4}
num_round = 100


new_bycourse = np.zeros(len(label_train))
new_bycourse_test = np.zeros(len(sample_test_x)-1)
print np.unique(courses_id)
for i in np.unique(courses_id):
    d_bycourse = data_train[data_train[0:,1]==i]
    l_bycourse = label_train[data_train[0:,1]==i]
    kf = KFold(len(l_bycourse),n_folds=10,shuffle=False)
    new_bycourse_i = np.zeros(len(l_bycourse))
    print "Number of users of courses"+str(i)+" is "+str(len(l_bycourse))
    for train_index, test_index in kf:
        d_bycourse_train = xgb.DMatrix( d_bycourse[train_index],label = l_bycourse[train_index] )
        d_bycourse_test = xgb.DMatrix( d_bycourse[test_index],label= l_bycourse[test_index] )
        #watchlist = [(d_bycourse_test,'eval'),(d_bycourse_train,'train')]
        #bst = xgb.train(param, d_bycourse_train ,num_round,watchlist)
        bst = xgb.train(param, d_bycourse_train ,num_round)
        preds = bst.predict(d_bycourse_test)
        new_bycourse_i[test_index] = preds
        print np.sum(new_bycourse_i!=0.0)
    new_bycourse[data_train[0:,1]==i] = new_bycourse_i
    print "Non-zero final pred is "+ str(np.sum(new_bycourse!=0.0))
    dMat_bycourse_train = xgb.DMatrix( d_bycourse, label=l_bycourse )
    test_d_bycourse = data_test[data_test[0:,1]==i]
    dummy = np.zeros(len(test_d_bycourse))
    dMat_bycourse_test = xgb.DMatrix( test_d_bycourse, label=dummy )
    #watchlist = [(dMat_bycourse_test,'eval'),(dMat_bycourse_train,'train')]
    bst = xgb.train(param, dMat_bycourse_train, num_round )
    pred_test = bst.predict(dMat_bycourse_test)
    new_bycourse_test[data_test[0:,1]==i] = pred_test
    

#tosave = np.hstack((enrollment_train[1:,0],new_bycourse))



f = open('bycoursefeat_train.csv','wb')
f.write('enrollment_id,bycourse_pred\n')
for i in range(0,len(sample_train_x)-1):
    f.write(str(sample_train_x[i+1,0])+','+str((new_bycourse[i]))+'\n')

f = open('bycoursefeat_test.csv','wb')
f.write('enrollment_id,bycourse_pred\n')
for i in range(0,len(sample_test_x)-1):
    f.write(str(sample_test_x[i+1,0])+','+str((new_bycourse_test[i]))+'\n')

'''
f = open('XGB_5_005_08_07_500_all_0110_cla_train.csv','wb')
for i in range(0,len(sample_train_x)-1):
    f.write(str(sample_train_x[i+1,0])+','+str(int(pred_train[i]))+'\n')
f = open('XGB_5_005_08_07_500_all_0110_cla_test.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str(int(pred_test[i]))+'\n')
'''

