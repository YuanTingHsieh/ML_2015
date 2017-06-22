#!/usr/bin/env python
# -*- coding: utf-8 -*-
import  numpy as np
import  pandas as pd
import  myparse as mp

from scipy import sparse
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder

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


sm = sparse.csr_matrix(np.vstack((data_train,data_test)))

wtff=100
#model = NMF(n_components=39)
model = TruncatedSVD(n_components=wtff)
alldata = model.fit_transform(sm)
new_data_train = alldata[0:len(label_train),0:]
new_data_test = alldata[len(label_train):,0:]

#print model.reconstruction_err_ 
print (model.explained_variance_ratio_.sum())

data_train = np.hstack((data_train,new_data_train))
data_test = np.hstack((data_test,new_data_test))
print np.shape(data_train)

np.savetxt("newfeatwithsvd0111_train.csv", new_data_train, delimiter=",",fmt="%s")
np.savetxt("newfeatwithsvd0111_test.csv", new_data_test, delimiter=",",fmt="%s")
