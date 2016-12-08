#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC

import time

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel = 'rbf', C = 10000)
time1 = time.time()
clf.fit(features_train, labels_train)
time2 = time.time()
predictions = clf.predict(features_test)
no_of_chris = predictions.sum()
time3 = time.time()
print "No Of Chris Classifications is ", no_of_chris
print "Time to Train ", round(time2 - time1, 3), " sec"
print "Time to Predict ", round(time3 - time2, 3), " sec"

#########################################################
