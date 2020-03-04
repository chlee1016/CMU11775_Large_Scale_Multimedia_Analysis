#!/bin/python 

#import numpy
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pandas as pd
import numpy as np

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1] 
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]

    ###########################################
    feature = pd.read_csv(feat_dir, header=None, delimiter = ' ')
    file_list = '/home/ubuntu/11775-hws/hw1_code/list/all.video'
    
    trn_list = '/home/ubuntu/11775-hws/all_trn.lst'
    c = pd.read_csv(trn_list, header=None, delimiter = ' ')
    # c = c.replace('P001',0)
    # c = c.replace('P002',1)
    # c = c.replace('P003',2)
    # c = c.fillna(4)
    # y_train = c.iloc[:,1]
    y_train = pd.get_dummies(c.iloc[:,1], dummy_na = True)
    # y_train = np.asarray(y_train)
    
    if event_name == 'P001':
        y_train = y_train.iloc[:,0]
    elif event_name == 'P002':
        y_train = y_train.iloc[:,1]
    else :
        y_train = y_train.iloc[:,2]
    
    
    trn_list = c.iloc[:,0]
    trn_list = np.asarray(trn_list)
    # print(trn_list)
    
    
    f = open(file_list, "r")
    idx = 0
    idx_list = []
    for line in f.readlines():
        
        if line.replace('\n','') in list(trn_list):
    #         print('yes')
            idx_list.append(idx)
        
        idx += 1
    f.close()
    
    x_train = feature.iloc[idx_list, :]
    
    
    print('x_train', x_train.shape[0], x_train.shape[1])
    print('y_train', y_train.shape[0])
    
    
    
    svm = SVC(C=0.001, gamma = 0.0001, probability=True)
    svm.fit(x_train, y_train)
    
    cPickle.dump(svm, open(output_file,"wb"), cPickle.HIGHEST_PROTOCOL)
    ##############################
    

    print 'SVM trained successfully for event %s!' % (event_name)
