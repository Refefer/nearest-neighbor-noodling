import sys
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score

import numpy as np

from boundaryforest import BoundaryForest
from pseudonn import *

def transform_feats(X, y):
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    return SelectFromModel(lsvc, prefit=True)

def main(train, test):
    train_X, train_y = load_svmlight_file(train)
    test_X, test_y = load_svmlight_file(test)

    #t = transform_feats(train_X, train_y)
    #train_X = t.transform(train_X).toarray()
    #test_X = t.transform(test_X).toarray()
    train_X = train_X.toarray()
    test_X = test_X.toarray()

    bf = BoundaryForest(50, 50)
    bf.fit(train_X, train_y)
    y_hat = bf.predict(test_X)
    print "BF:", accuracy_score(y_hat, test_y)

    #knn = KNeighborsClassifier()
    #knn.fit(train_X, train_y)
    #y_hat = knn.predict(test_X)
    #print "KNN:", accuracy_score(y_hat, test_y)
 
    #clf = LogisticRegression(n_jobs=-1)
    #clf.fit(train_X, train_y)
    #y_hat = clf.predict(test_X)
    #print "LR:", accuracy_score(y_hat, test_y)

    #clf = PseudoNN(HartTrainer(NoopTrainer(), outliers=False), 1)
    #clf = PseudoNN(KmeansTrainer(20), 1)
    #clf.fit(train_X, train_y)
    #y_hat = clf.predict(test_X)
    #print "PNN:", accuracy_score(y_hat, test_y)

if __name__ == '__main__':
    main(*sys.argv[1:])
