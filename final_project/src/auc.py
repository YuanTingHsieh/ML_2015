#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import myparse as mp

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score


mypred = mp.readcsv("XGB_5_005_08_07_450_all0111_reg_train968152.csv").astype(float)
truth_train = mp.readcsv("truth_train.csv").astype(float)

mypred[mypred<0.5]=0
mypred[mypred>=0.5]=1

print float(sum(truth_train[0:,1]==mypred[0:,1]))/len(mypred[0:,1])
print roc_auc_score(truth_train[0:,1].astype(float),mypred[0:,1].astype(float))
