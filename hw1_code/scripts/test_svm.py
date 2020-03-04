#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import cPickle
import sys
import pandas as pd

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    event_name = sys.argv[5]
    ##################################
    feature = pd.read_csv(feat_dir, header=None, delimiter = ' ')
    file_list = 'list/all.video'

    svm = cPickle.load(open(model_file,"rb"))
    
    ############
    # case 1. validation
    val_list = '/home/ubuntu/11775-hws/all_val.lst'
    d = pd.read_csv(val_list, header=None, delimiter = ' ')
    y_val = pd.get_dummies(d.iloc[:,1], dummy_na = True)

    
    if event_name == 'P001':
        y_val = y_val.iloc[:,0]
    elif event_name == 'P002':
        y_val = y_val.iloc[:,1]
    else :
        y_val = y_val.iloc[:,2]

    val_list = d.iloc[:,0]
    val_list = np.asarray(val_list)

    f = open(file_list, "r")
    idx = 0
    idx_list = []
    for line in f.readlines():
        
        if line.replace('\n','') in list(val_list):
            idx_list.append(idx)
        
        idx += 1
    f.close()
    
    x_val = feature.iloc[idx_list, :]

    print('x_val', x_val.shape[0], x_val.shape[1])
    print('y_val', y_val.shape[0])
    
    predicted_proba = svm.predict_proba(x_val)
    from sklearn.metrics import average_precision_score
    print(event_name,' ',average_precision_score(y_val, predicted_proba[:,1]))

    ############
    # case 2. test
    test_list = '/home/ubuntu/11775-hws/all_test_fake.lst'
    e = pd.read_csv(test_list, header=None, delimiter = ' ')
    test_list = e.iloc[:,0]
    test_list = np.asarray(test_list)

    
    f = open(file_list, "r")
    idx = 0
    idx_list = []
    for line in f.readlines():
        
        if line.replace('\n','') in list(test_list):
            idx_list.append(idx)
        
        idx += 1
    f.close()
    
    x_test = feature.iloc[idx_list, :]

    print('x_test', x_test.shape[0], x_test.shape[1])
    
    predicted_proba = svm.predict_proba(x_test)
    np.savetxt(event_name + '_MFCC.lst', predicted_proba[:,1], delimiter = " ")
    ##################################
    
