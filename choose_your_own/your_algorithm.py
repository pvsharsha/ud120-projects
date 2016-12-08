#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#clf = KNeighborsClassifier(n_neighbors = 2, weights='distance', n_jobs = -1)
#clf.fit(features_train, labels_train)
#acc = clf.score(features_test, labels_test)
#print("Accuracy is %.2f" % acc)


#clf = RandomForestClassifier(n_estimators = 10)
#clf.fit(features_train, labels_train)
#acc = clf.score(features_test, labels_test)
#print("Accuracy is %.2f" % acc)


#clf = AdaBoostClassifier(n_estimators = 1000)
#clf.fit(features_train, labels_train)
#acc = clf.score(features_test, labels_test)
#print("Accuracy is %.2f" % acc)


classifiers = {
            'k-NN' : KNeighborsClassifier(n_neighbors = 2, weights='distance', n_jobs = -1),
            'Random-Forest' : RandomForestClassifier(n_estimators = 20),
            'AdaBoost' : AdaBoostClassifier(n_estimators = 100),
            'naive_bayes' :  GaussianNB(),
            'svm' : SVC(kernel = 'rbf', C = 10000)
        }

for name in classifiers:
    clf = classifiers[name]
    clf.fit(features_train, labels_train)
    acc_train = clf.score(features_train, labels_train)
    acc_test = clf.score(features_test, labels_test)
    print("Accuracy for %s for train is %.4f and test is %.4f" % (name, acc_train, acc_test))
    try:
        prettyPicture(clf, features_test, labels_test, name)
    except NameError:
        pass
