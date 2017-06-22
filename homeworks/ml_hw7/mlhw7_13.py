#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import csv
import math

def readcsv(input_file):
    print "Reading "+input_file
    with open(input_file,'rb') as csvfile:
        data_iter = csv.reader(csvfile,delimiter=' ')
        data = [data for data in data_iter]
    data_array = np.asarray(data).astype(float)
    return data_array

def comp_gini(pos_left,neg_left,pos_right,neg_right):
    left_tol = pos_left+neg_left
    right_tol = pos_right+neg_right

    if (left_tol==0):
        left_gini =0
    else:
        left_gini = 1-math.pow ((float(pos_left)/left_tol),2)-math.pow((float(neg_left)/left_tol),2)
    if (right_tol==0):
        right_gini=0 
    else:
        right_gini = 1-math.pow( (float(pos_right)/right_tol),2)-math.pow((float(neg_right)/right_tol),2)
    return float(left_tol)/(left_tol+right_tol)*left_gini+float(right_tol)/(left_tol+right_tol)*right_gini

def DecisionTree(data,cand_attributes,height=0):
    class_ind = 2
    #One leaf
    if (len(data)==1):
        return data[class_ind]
    #All same classes
    allpos = sum(data[0:,class_ind]==1)
    if (allpos==0):
       return -1
    elif (allpos==len(data)):
       return 1
    if (len(cand_attributes)==0):
        return np.sign(sum(data[0:,class_ind]))
    best_gini_split = np.inf
    best_sign_split = np.inf
    best_theta_split = np.inf
    best_attr_split = np.inf
    best_par_split = np.inf
    for i in cand_attributes:
        index = np.argsort(data[0:,i])
        sorted_data = data[index,0:]
        sorted_fea = sorted_data[0:,i]
        middles = sorted_fea[:-1]+np.diff(sorted_fea)/2
        middles = np.append(-np.inf,middles)
        middles = np.append(middles,np.inf)
        best_attr_gini = np.inf
        best_attr_sign = 0
        best_attr_theta = 0
        best_attr_par = 0
        pos_left = 0
        neg_left = 0
        pos_right = allpos
        neg_right = len(data)-allpos
        
        for j in range(0,len(middles)):
            sign_now = 1 
            gini_now = comp_gini(pos_left,neg_left,pos_right,neg_right)
            #gini_now_sneg = comp_gini(neg_left,pos_left,neg_right,pos_right)
            #print j
            #print gini_now

            #print gini_now_sneg
            #if gini_now_sneg < gini_now:
            #   gini_now = gini_now_sneg
            #   sign_now = -1 
            if gini_now < best_attr_gini:
               best_attr_gini = gini_now
               best_attr_sign = sign_now
               best_attr_theta = middles[j]
               best_attr_par = j
            if j!=len(middles)-1:
                if sorted_data[j,class_ind]==1:
                    pos_left = pos_left +1
                    pos_right = pos_right -1
                else:
                    neg_left = neg_left+1
                    neg_right = neg_right -1
            

        if best_attr_gini < best_gini_split:
            best_sign_split = best_attr_sign 
            best_gini_split = best_attr_gini
            best_theta_split = best_attr_theta 
            best_attr_split = i
            best_par_split = best_attr_par
        
    treenode = {'attr':best_attr_split,'theta':best_theta_split,'sign':best_sign_split,'height':height} 
    #print treenode
    #print best_par_split
    last_index = np.argsort(data[0:,best_attr_split])
    sorted_data = data[last_index,0:]
    left_data = sorted_data[0:best_par_split,0:]
    right_data = sorted_data[best_par_split:,0:]
    remain_cand_attributes = [i for i in cand_attributes if i != best_attr_split]
    
    treenode['lefttree']=DecisionTree(left_data,remain_cand_attributes,height+1)
    treenode['righttree']=DecisionTree(right_data,remain_cand_attributes,height+1)
    return treenode


#train_all = readcsv("test.dat")
train_all = readcsv("hw7_train.dat")
test_all = readcsv("hw7_test.dat")


oop = DecisionTree(train_all,range(2),0)
print oop