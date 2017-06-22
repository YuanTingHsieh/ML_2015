"""Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.
"""

from __future__ import division
import numpy as np
#import load_data
from sklearn.cross_validation import StratifiedKFold,train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression

def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt))


if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = True

    #X, y, X_submission = load_data.load()
    feat1 = np.genfromtxt('feat_train.csv',delimiter=',')[:,1:]
    #feat2 = np.genfromtxt('augmentGraph_train.csv',delimiter=',')[1:,1:]
    
    #X = feat1#np.append(feat1,feat2,axis=1)
    truth = np.genfromtxt('truth_train.csv',delimiter=',')[:,1]
    print(truth)
    X_submission=np.genfromtxt('feat_test.csv',delimiter=',')[:,1:]
    
    X,X_submission,y,y_truth = train_test_split(feat1,truth,test_size=0.2) 
    print(y)
    X_new =X
    y_new =y
    count =0
    if shuffle:
        idx = np.random.permutation(y.size)
        #print(y.size,y.shape)
        X = X[idx]
        y= y[idx]
        #count += 1 
    #print(X[0,:])    
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=1000,max_depth=19, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=1000,max_depth=19, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=140,max_depth=19, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=140,max_depth=19, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=100)]
    
    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            #lr = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
            #lr.fit(X_train,y_train) 
            clf.fit(X_train, y_train)
            
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
            print(clf.score(X_test,y_test))
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    #np.savetxt('blend_train1.csv',dataset_blend_train,delimiter =',')
    #np.savetxt('y_train1.csv',y,delimiter=',')
    #np.savetxt('y_test1.csv',y_truth,delimiter=',') 
    #np.savetxt('blend_test1.csv',dataset_blend_test,delimiter=',')
    print(dataset_blend_test)
    print "Blending."
    #clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
    clf  = RandomForestClassifier(n_jobs =-1,n_estimators=1000,max_depth =5,oob_score=True)
    clf.fit(dataset_blend_train, y)
    print(clf.oob_score_)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print "Saving Results."
    #np.savetxt(fname='test.csv', X=y_submission, fmt='%0.9f')
    print(y_submission)
    y_pred = (np.sign(y_submission-0.5)+1)/2
    #np.savetxt(fname='test_pred.csv', X=y_pred, fmt='%0.9f')
    err_mat = y_pred-y_truth
    err =np.sum(np.abs(err_mat))/(y_truth.shape[0])
    print(1-err)
    
