from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np

class Trainer(object):

    def train(self, X, y, rs):
        raise NotImplementedError()

    def get_cluster_size(self, X_set):
        # Get clusters
        if self.n_clusters_per_class < 1:
            X_size = X_set.shape[0]
            k = max(1, int(self.n_clusters_per_class * X_size))
        else:
            k = self.n_clusters_per_class

        return min(k, X_set.shape[0])

    def class_iter(self, X, y):
        classes = set(y)
        new_X, new_y = [], []
        for yi in classes:
            x_subset = X[np.where(y == yi)]
            yield x_subset, yi

class NoopTrainer(Trainer):
    def train(self, X, y, rs):
        return X, y

class KmeansTrainer(Trainer):
    def __init__(self, n_clusters_per_class):
        self.n_clusters_per_class = n_clusters_per_class

    def train(self, X, y, rs):
        new_X, new_y = [], []
        for x_sub, yi in self.class_iter(X, y):
            clusters = self.get_cluster_size(x_sub)
            clf = KMeans(clusters, random_state=rs)
            clf.fit(x_sub)
            new_X.append(clf.cluster_centers_)
            new_y.extend([yi] * new_X[-1].shape[0])

        return np.vstack(new_X), new_y

class RandomTrainer(Trainer):
    def __init__(self, n_clusters_per_class=5, retries=1, nn=3):
        self.n_clusters_per_class = n_clusters_per_class
        self.retries = retries
        self.nn = 1

    def train(self, X, y, rs):
        best = (float('-inf'), None)
        for _ in range(self.retries):
            nX, ny = self._train(X, y, rs)
            clf = KNeighborsClassifier(self.nn)
            clf.fit(nX, ny)
            score = clf.score(X, y)
            if score > best[0]:
                best = (score, (nX, ny))

        return best[1]
    
    def _train(self, X, y, rs):
        new_X, new_y = [], []
        for x_sub, yi in self.class_iter(X, y):
            clusters = self.get_cluster_size(x_sub)
            # Choose random clusters
            idxs = rs.choice(range(x_sub.shape[0]), clusters)
            new_X.append(x_sub[idxs])
            new_y.extend([yi] * new_X[-1].shape[0])

        return np.vstack(new_X), new_y

class KmeansPlusTrainer(Trainer):
    def __init__(self, n_clusters_per_class=5, retries=1, nn=1):
        self.n_clusters_per_class = n_clusters_per_class
        self.retries = retries
        self.nn = nn

    def train(self, X, y, rs):
        best = (float('-inf'), None)
        for _ in range(self.retries):
            nX, ny = self._train(X, y, rs)
            clf = KNeighborsClassifier(self.nn)
            clf.fit(nX, ny)
            score = clf.score(X, y)
            if score > best[0]:
                best = (score, (nX, ny))

        return best[1]
    
    def _train(self, X, y, rs):
        new_X, new_y = [], []
        for x_sub, yi in self.class_iter(X, y):
            clusters = self.get_cluster_size(x_sub)
            # Choose random clusters
            centers = _init_centroids(x_sub, clusters, 'k-means++', rs)
            new_X.append(centers)
            new_y.extend([yi] * new_X[-1].shape[0])

        return np.vstack(new_X), new_y

class HartTrainer(Trainer):
    def __init__(self, trainer, iters=1, passes=1, outliers=True):
        self.trainer = trainer
        self.iters = iters
        self.passes = passes
        self.outliers = outliers

    def remove_outliers(self, X, y, dist):
        indices = set(range(X.shape[0]))
        to_remove = set()
        nX, ny = [], []
        for i in indices:
            nearest = np.argmin(dist[i])
            for nidx in np.argsort(dist[i]):
                if nidx not in to_remove:
                    if y[nidx] != y[i]:
                        to_remove.add(i)
                    else:
                        nX.append(X[i])
                        ny.append(y[i])

                    break

        return np.vstack(nX), np.array(ny)

    def compute_dists(self, X):
        dist = euclidean_distances(X, X)
        for i in range(X.shape[0]):
            dist[i,i] = np.inf

        return dist

    def collapse(self, X, y, rs):
        if not self.outliers:
            dist = self.compute_dists(X)

            # Remove outliers
            X, y = self.remove_outliers(X, y, dist)

        dist = self.compute_dists(X)
        
        indices = range(X.shape[0])

        done = False
        it = 0
        while not done and it != self.iters:
            rs.shuffle(indices)
            it += 1
            # seed new set with random point
            new_idxs = set([rs.choice(indices)])

            # iterate through points, adding new ones
            for _ in range(self.passes):
                for idx in indices:
                    uidx = min((dist[idx, uidx], uidx) for uidx in new_idxs)[1]
                    if y[idx] != y[uidx]:
                        new_idxs.add(idx)

            done = len(new_idxs) == len(indices)
            indices = list(new_idxs)

        return X[indices], y[indices]

    def train(self, X, y, rs):
        uX, uy = self.collapse(X, y, rs)
        return self.trainer.train(uX, uy, rs)

class PseudoNN(object):
    estimator_ = None
    def __init__(self, trainer, nn=1, random_state=2019):
        self.trainer = trainer
        self.nn = nn
        self.seed = random_state

    def fit(self, X, y):
        rs = np.random.RandomState(self.seed)
        new_X, new_y = self.trainer.train(X, y, rs)
        clf = KNeighborsClassifier(min(self.nn, new_X.shape[0] - 1))
        clf.fit(new_X, new_y)
        self.estimator_ = clf

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def score(self, X, y):
        return self.estimator_.score(X, y)

