import numpy as np
import myparse as mp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

label_train = mp.readcsv('truth_train.csv')[0:,1].astype(float)

feat1_train = mp.readcsv('XGB_5_005_08_07_450_all0111_reg_train968152.csv')[0:,1].astype(float)
feat1_test = mp.readcsv('XGB_5_005_08_07_450_all0111_reg_test968152.csv')[0:,1].astype(float)

feat2_train = mp.readcsv('XGB_5_005_08_07_500_all0110_reg_train967211.csv')[0:,1].astype(float)
feat2_test = mp.readcsv('XGB_5_005_08_07_500_all0110_reg_test967211.csv')[0:,1].astype(float)

feat3_train = mp.readcsv('XGB_5_005_08_07_400_all0110_reg_train967298.csv')[0:,1].astype(float)
feat3_test = mp.readcsv('XGB_5_005_08_07_400_all0110_reg_test967298.csv')[0:,1].astype(float)

feat4_train =mp.readcsv('XGB_5_005_08_07_300_all0111_reg_rank_train967192.csv')[0:,1].astype(float)
feat4_test =mp.readcsv('XGB_5_005_08_07_300_all0111_reg_rank_test967192.csv')[0:,1].astype(float)

feat5_train = mp.readcsv('XGB_5_005_08_09_500_all_0111_impor_reg_train966810.csv')[0:,1].astype(float)
feat5_test = mp.readcsv('XGB_5_005_08_09_500_all_0111_impor_reg_test966810.csv')[0:,1].astype(float)

'''
wtf = np.multiply(feat1_train,feat2_train)
wtf = np.multiply(feat3_train,wtf)
wtf = np.multiply(feat4_train,wtf)
wtf = np.multiply(feat5_train,wtf)

wtff =np.multiply(feat1_test,feat2_test)
wtff =np.multiply(feat3_test,wtff)
wtff =np.multiply(feat4_test,wtff)
wtff =np.multiply(feat5_test,wtff)

thecount = np.zeros(len(feat1_test))
thecount[np.argsort(feat1_test)[0:18000]]+=1
thecount[np.argsort(feat2_test)[0:18000]]+=float(967211)/968152
thecount[np.argsort(feat3_test)[0:18000]]+=float(967298)/968152
thecount[np.argsort(feat4_test)[0:18000]]+=float(967192)/968152
thecount[np.argsort(feat5_test)[0:18000]]+=float(966810)/968152

final = np.argsort(thecount)

plt.plot(thecount)
plt.show()

highest_seq = np.argsort(feat1_test)

new_seq = np.argsort(wtff)
print highest_seq
print new_seq

print len(feat1_test)
print sum(highest_seq[0:18000]!=final[0:18000])
print sum(new_seq[0:18000]!=final[0:18000])
'''

allfeat_train = np.vstack((feat1_train,feat2_train))
allfeat_train = np.vstack((allfeat_train,feat3_train))
allfeat_train = np.vstack((allfeat_train,feat4_train))
allfeat_train = np.vstack((allfeat_train,feat5_train))

allfeat_test = np.vstack((feat1_test,feat2_test))
allfeat_test = np.vstack((allfeat_test,feat3_test))
allfeat_test = np.vstack((allfeat_test,feat4_test))
allfeat_test = np.vstack((allfeat_test,feat5_test))

allfeat_train = np.transpose(allfeat_train)
allfeat_test = np.transpose(allfeat_test)

dtrain = xgb.DMatrix(allfeat_train,label=label_train)
param = {'max_depth':4, 'eta':0.05, 'silent':1, 'objective':'binary:logistic','subsample':0.8,'colsample_bytree':1.0,'nthread':4}
num_round = 300

bst = xgb.train(param,dtrain,num_round)

dummy = np.zeros(np.shape(data_test)[0])
dtest = xgb.DMatrix(data_test,label= dummy)
pred_train = bst.predict(dtrain)
#pred_train[pred_train>0.5]=1
#pred_train[pred_train<=0.5]=0
#cor =  (len(label_train)-sum(abs(pred_train-label_train)))/len(label_train)
cor = roc_auc_score(label_train,pred_train)
print 'AUC = %0.6f' % cor
pred_test = bst.predict(dtest)
#pred_test[pred_test>0.5]=1
#pred_test[pred_test<=0.5]=0
#print np.shape(pred_test)