from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

class Node(object):
    def __init__(self, idx):
        self.idx = idx
        self.children = []

class BoundaryTree(object):
    def __init__(self, table, k, distance, c_metric=lambda x, y: x == y):
        self.k = k
        self.distance = distance
        self.c_metric = c_metric

        self.root = None
        self.table = table 

    def _score(self, idx, q):
        return self.distance(self.table[idx][0], q)

    def _traverse(self, y):
        node = self.root
        node_score = self._score(node.idx, y)
        done = False
        while not done:
            if len(node.children) > 0:
                best = min((self._score(vn.idx, y), vn) for vn in node.children)
            else:
                best = (float('inf'), None)

            # If we're less than K, add the intermediate node
            if len(node.children) < self.k:
                best = min((node_score, node), best)

            # Are we done?
            done = best[1].idx == node.idx
            node_score, node = best

        return node_score, node

    def insert(self, idx):
        if self.root is None:
            self.root = Node(idx)
            return True
        
        y, cy = self.table[idx]
        # Find the closest candidate
        node_score, node = self._traverse(y)
        if not self.c_metric(self.table[node.idx][1], cy):
            # Add the node
            new_node = Node(idx)
            node.children.append(new_node)
            return True

        return False

    def query(self, y):
        node_score, node = self._traverse(y)
        return node_score, self.table[node.idx][1]

def sq_euc_dist(x, y):
    return ((x - y) ** 2).sum()

def shepard_classification(scores, dists, raw=False):
    classes = defaultdict(float)
    denom = 0.0
    for dist, cls in scores:
        classes[cls] += 1. / (dist + 1e-6)
        denom += 1. / (dist + 1e-6)

    score, cls = max((v / denom, k) for k, v in classes.iteritems())
    if raw:
        return (cls, score)

    return cls

class BoundaryForest(object):
    def __init__(self, n_trees, k, 
            distance = sq_euc_dist, 
            c_metric = lambda x, y: x == y, 
            seed     = 2018,
            estimator = shepard_classification):

        self.k = k
        self.n_trees = n_trees
        self.distance = distance 
        self.c_metric = c_metric
        self.estimator = estimator
        self.seed = seed

    def _add(self, y, cy):
        self.table[self.nodes_cnt] = (y, cy)
        self.nodes_cnt += 1
        return self.nodes_cnt - 1

    def _init(self):
        self.table = {}
        self.nodes_cnt = 0
        self.trees = [BoundaryTree(self.table, self.k, self.distance, self.c_metric) 
                for _ in range(self.n_trees)]
        self.rs = np.random.RandomState(self.seed)

    def insert(self, xi, yi):
        idx = self._add(xi, yi)
        added = False
        for t in self.trees:
            added |= t.insert(idx)

        if not added:
            del self.table[idx]

    def partial_fit(self, X, y):
        for i in range(X.shape[0]):
            self.insert(X[i], y[i])

    def fit(self, X, y):
        self._init()
        
        # Add all to table
        new_idxs = []
        for i in range(X.shape[0]):
            xi, yi = X[i], y[i]
            new_idxs.append(self._add(xi, yi))

        added_idxs = set()
        # Smart seed
        subset = new_idxs[:len(self.trees)]
        for i, t in enumerate(self.trees):
            self.rs.shuffle(subset)
            for idx in subset:
                if t.insert(idx):
                    added_idxs.add(idx)

        # Remove superfluous nodes
        for r_idx in set(new_idxs) - added_idxs:
            del self.table[r_idx]

        self.partial_fit(X, y)

    def predict(self, X):
        y_hat = []
        for xi in X:
            scores = [t.query(xi) for t in self.trees]
            y_hat.append(self.estimator(scores, False))

        return np.vstack(y_hat)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
