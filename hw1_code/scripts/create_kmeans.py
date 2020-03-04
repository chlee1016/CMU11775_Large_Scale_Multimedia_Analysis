#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    ###########################################
    kmeans_feature_list = []

    f = open(file_list, "r")
    for line in f.readlines():
        mfcc_path = "mfcc/" + line.replace('\n','') + ".mfcc.csv"
        #f_kmeans_write = open('kmeans/' + line.replace('\n',''), 'w')
        
        
        if os.path.exists(mfcc_path) == True:
            a = pd.read_csv(mfcc_path, header = None, delimiter = ";") 
            a = np.asarray(a)
            kmeans_predicted = kmeans.predict(a)
            kmeans_predicted = np.eye(cluster_num)[kmeans_predicted]
            kmeans_predicted = np.sum(kmeans_predicted, axis = 0)
            print(kmeans_predicted.shape)
            
            
    
        else:
            print('The file ' + line.replace('\n','') + ' does not exist in the list')
            kmeans_predicted = np.zeros(cluster_num)
    
        #f_kmeans_wirte.write(line + '\n')
        #f_kmeans_write.write.close()
        kmeans_feature_list.append(kmeans_predicted)
        np.savetxt('kmeans/' + line.replace('\n',''), kmeans_predicted.reshape(1,-1), delimiter = " ")
        
    np.savetxt('kmeans_feature', np.asarray(kmeans_feature_list), delimiter = " ")
    f.close()
    ###########################################
    print "K-means features generated successfully!"




echo "Creating k-means cluster vectors"
python2 scripts/create_kmeans.py kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;
