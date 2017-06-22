#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import myparse as mp

#aaa = "sample_test_x.csv"
#bbb = "XGB_5_005_08_09_400_all_0111_cla_single_test.csv"
#ccc = "XGB_5_005_08_07_500_all_0110_cla_test884551.csv"
aaa = "sample_train_x.csv"
bbb = "XGB_5_005_08_09_350_all_0111_cla_single_train.csv"
ccc = "XGB_5_005_08_07_500_all_0110_cla_train884551.csv"

truth_train = mp.readcsv("truth_train.csv")
label_train = truth_train[0:,1].astype(float)

sample_test_x = mp.readcsv(aaa)
data_test = sample_test_x[1:,1:].astype(float)
print sample_test_x[0,3]
print sample_test_x[0:6,3]
print data_test[0:5,2]

index_test_single = data_test[0:,2]==1
index_test_multiple = data_test[0:,2]!=1

print sum(index_test_single)
print sum(index_test_multiple)

single_pred = mp.readcsv(bbb)[0:,1].astype(float)
nice_pred = mp.readcsv(ccc)[0:,1].astype(float)

merge_pred = np.zeros(len(single_pred))
merge_pred[index_test_single] = single_pred[index_test_single]
merge_pred[index_test_multiple] = nice_pred[index_test_multiple]

print len(label_train[index_test_single])
print len(single_pred[index_test_single])

print 'Single of single accu is '+str(float(sum(label_train[index_test_single]==single_pred[index_test_single]))/len(label_train[index_test_single]))
print 'Multi of single accu is '+str(float(sum(label_train[index_test_multiple]==single_pred[index_test_multiple]))/len(label_train[index_test_multiple]))
print 'Total of single accu is '+str(float(sum(label_train==single_pred))/len(label_train))

print 'Single of nice accu is '+str(float(sum(label_train[index_test_single]==nice_pred[index_test_single]))/len(label_train[index_test_single]))
print 'Multi of nice accu is '+str(float(sum(label_train[index_test_multiple]==nice_pred[index_test_multiple]))/len(label_train[index_test_multiple]))
print 'Total of nice accu is '+str(float(sum(label_train==nice_pred))/len(label_train))

print 'Total of merged accu is' + str(float(sum(label_train==merge_pred))/len(label_train))

#f = open('XGB_0111_impor_merge_train.csv','wb')
#for i in range(0,len(pred_train)):
#    f.write(str(sample_train_x[i+1,0])+','+str((pred_train[i]))+'\n')
#f = open('XGB_0111_merge_400_cla_test.csv','wb')
#for i in range(0,len(merge_pred)):
#    f.write(str(sample_test_x[i+1,0])+','+str(int(merge_pred[i]))+'\n')
