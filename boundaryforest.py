from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

class DistMetric(object):
    def compare(self, x, y):
        raise NotImplementedError()

    def batch_compare(self, x, ys):
        return [self.compare(x, y) for y in ys]

class LambdaMetric(DistMetric):
    def __init__(self, f):
        self.f = f

    def compare(self, x, y):
        return self.f(x, y)

class EucDist(DistMetric):
    def compare(self, x, y):
        return ((x - y) ** 2).sum()

class SparseEucDist(DistMetric):
    def compare(self, x, y):
        return (x - y).power(2).sum()

class LabelDistance(object):
    def compare(self, cx, cy):
        raise NotImplementedError()

class EqualDist(LabelDistance):
    def compare(self, cx, cy):
        return cx == cy

class ThresholdDist(LabelDistance):
    def __init__(self, threshold):
        self.threshold = threshold

    def compare(self, cx, cy):
        return abs(cx - cy) < self.threshold

class Estimator(object):
    def scorer(self, scores):
        raise NotImplementedError()
            
class ShepardClassifier(Estimator):
    def score(self, scores):
        classes = defaultdict(float)
        denom = 0.0
        for dist, cls in scores:
            weight = 1. / (dist + 1e-6)
            classes[cls] += weight
            denom += weight

        score, cls = max((v / denom, k) for k, v in classes.iteritems())
        return cls, score

class ShepardRegressor(Estimator):
    def score(self, scores):
        score = 0.0
        denom = 0.0
        for dist, cls in scores:
            weight = 1. / (dist + 1e-6)
            score += cls * weight
            denom += weight

        return score / denom, 1

class UniformRegressor(Estimator):
    def score(self, scores):
        return sum(cls for _, cls in scores) / float(len(scores)), 1

class Node(object):
    def __init__(self, idx):
        self.idx = idx
        self.children = []

class BoundaryTree(object):
    def __init__(self, table, k, distance, label_distance):
        self.k = k
        self.distance = distance
        self.label_distance = label_distance

        self.root = None
        self.table = table 

    def _score(self, q, idxs):
        xs = [self.table[idx][0] for idx in idxs]
        return self.distance.batch_compare(q, xs)

    def _traverse(self, y):
        node = self.root
        node_score = self._score(y, [node.idx])[0]
        done = False
        while not done:
            if len(node.children) > 0:
                c_scores = self._score(y, (vn.idx for vn in node.children))
                best = min((c_scores[i], vn) for i, vn in enumerate(node.children))
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
        if not self.label_distance.compare(self.table[node.idx][1], cy):
            # Add the node
            new_node = Node(idx)
            node.children.append(new_node)
            return True

        return False

    def query(self, y):
        node_score, node = self._traverse(y)
        return node_score, self.table[node.idx][1]

class BoundaryForest(object):
    def __init__(self, n_trees, k, 
            distance       = EucDist(), 
            label_distance = EqualDist(),
            estimator      = ShepardClassifier(),
            seed           = 2018):

        if not isinstance(distance, DistMetric):
            if callable(distance):
                distance = LambdaMetric(distance)
            else:
                raise AssertionError("Distance isn't a DistMetric or callable!")

        self.k = k
        self.n_trees = n_trees
        self.distance = distance 
        self.label_distance = label_distance
        self.estimator = estimator
        self.seed = seed

    def _add(self, y, cy):
        self.table[self.nodes_cnt] = (y, cy)
        self.nodes_cnt += 1
        return self.nodes_cnt - 1

    def _init(self):
        self.table = {}
        self.nodes_cnt = 0
        self.trees = [BoundaryTree(self.table, self.k, self.distance, self.label_distance) 
                for _ in range(self.n_trees)]
        self.rs = np.random.RandomState(self.seed)

    def insert(self, xi, yi):
        idx = self._add(xi, yi)
        added = False
        for t in self.trees:
            added |= t.insert(idx)

        if not added:
            del self.table[idx]

    def partial_fit(self, X, y, offset=0):
        if not hasattr(self, 'trees'):
            self._init()

        for i in range(offset, X.shape[0]):
            self.insert(X[i], y[i])

    def fit(self, X, y):
        self._init()
        
        # Add all to table
        subset = []
        for i in range(min(len(self.trees), X.shape[0])):
            subset.append(self._add(X[i], y[i]))

        added_idxs = set()
        # Smart seed
        for i, t in enumerate(self.trees):
            self.rs.shuffle(subset)
            for idx in subset:
                if t.insert(idx):
                    added_idxs.add(idx)

        # Remove superfluous nodes
        for r_idx in set(subset) - added_idxs:
            del self.table[r_idx]
            
        self.partial_fit(X, y, offset=len(subset))

        return self

    def predict(self, X):
        y_hat = []
        for xi in X:
            scores = [t.query(xi) for t in self.trees]
            y_hat.append(self.estimator.score(scores)[0])

        return np.vstack(y_hat)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def BFClassifier(n_trees, k, **kwargs):
    return BoundaryForest(n_trees, k, **kwargs)

def BFRegressor(n_trees, k, threshold, weight='distance'):

    assert weight in ('distance', 'uniform')
    if weight == 'distance':
        estimator = ShepardRegressor()
    else:
        estimator = UniformRegressor()

    label_distance = ThresholdDist(threshold)
    return BoundaryForest(n_trees, k, 
            label_distance=label_distance, 
            estimator=estimator)

