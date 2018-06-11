from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, make_moons, make_circles, make_classification, load_iris
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

import numpy as np

from pseudonn import *
from boundaryforest import BoundaryForest

def mag(v):
    return (v ** 2).sum() ** .5

def cosine_similarity(x, y):
    return 1 - (x.dot(y) / (mag(x) * mag(y)))
        
def main():
    N = 10000
    datasets = [
        load_digits(return_X_y=True),
        make_moons(N, noise=0.3, random_state=2016),
        make_circles(N, noise=0.2, factor=0.5, random_state=2017),
        load_iris(return_X_y=True),
        make_classification(N, n_features=200, n_informative=5,
                        random_state=1, n_classes=5, n_clusters_per_class=3)
    ]

    for i, (X, y) in enumerate(datasets):
        print "\nStart {}".format(i)
        rs = np.random.RandomState(2018)

        idxs = range(X.shape[0])
        rs.shuffle(idxs)
        test_size = int(X.shape[0] * .1)
        test_idx, train_idx = idxs[:test_size], idxs[test_size:]

        tr_X, tr_y = X[train_idx], y[train_idx]
        tst_X, tst_y = X[test_idx], y[test_idx]
        print "X:", tr_X.shape[0]

        for clf in [
                NearestCentroid(),
                BaggingClassifier(NearestCentroid(), max_samples=0.2, random_state=2022),
                LogisticRegression(),
                KNeighborsClassifier(5),
                GaussianNB(),
                PseudoNN(KmeansTrainer(20), 2),
                PseudoNN(RandomTrainer(20, retries=10), 2),
                PseudoNN(KmeansPlusTrainer(20, retries=10, nn=1), 2),
                PseudoNN(HartTrainer(NoopTrainer()), 1),
                BoundaryForest(50, 50, metric=cosine_similarity)
            ]:
            clf.fit(tr_X, tr_y)

            y_hat = clf.predict(tst_X)
            if hasattr(clf, 'base_estimator'):
                name = type(clf.base_estimator).__name__
            else:
                name = type(clf).__name__

            print name, f1_score(tst_y, y_hat, average='micro')

if __name__ == '__main__':
    main()
