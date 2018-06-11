from sklearn.datasets import load_digits, make_moons, make_circles, make_classification

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pseudonn import *
from boundaryforest import *
import numpy as np

ROWS = 6
COLS = 2
def get_ax(idx):
    return plt.subplot(ROWS, COLS, idx)

rs = np.random.RandomState(1028)
X, y = make_moons(1000, noise=0.3, random_state=2016)
#X, y = make_circles(1000, noise=0.2, factor=0.5, random_state=2017)
#X, y = load_digits(return_X_y=True)

figure = plt.figure(figsize=(8, 30))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
# Plot the training points
ax = get_ax(1)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
           edgecolors='k')

for i in range(1, 4):
    ht = HartTrainer(NoopTrainer(), 1, 3)
    uX, uy = ht.collapse(X, y, rs)
    ax = get_ax(2 * i + 1)
    ax.scatter(uX[:, 0], uX[:, 1], c=uy, cmap=cm_bright,
               edgecolors='k')
    uX, uy = ht.train(X, y, rs)
    print "Hart:", len(uX)
    ax = get_ax(2 * i + 2)
    ax.scatter(uX[:, 0], uX[:, 1], c=uy, cmap=cm_bright,
               edgecolors='k')

# Random Trainer
uX, uy = RandomTrainer(10).train(X, y, rs)
ax = get_ax(ROWS*COLS - 3)
ax.scatter(uX[:, 0], uX[:, 1], c=uy, cmap=cm_bright, edgecolors='k')

# KMeans Trainer
uX, uy = KmeansTrainer(10).train(X, y, rs)
ax = get_ax(ROWS*COLS - 2)
ax.scatter(uX[:, 0], uX[:, 1], c=uy, cmap=cm_bright, edgecolors='k')

# bf Trainer
bt = BoundaryForest(10, 50, sq_euc_dist)
bt.fit(X, y)
uX, uy = zip(*bt.table.values())
uX = np.vstack(uX)
print "BF:", len(uX)
ax = get_ax(ROWS*COLS - 1)
ax.scatter(uX[:, 0], uX[:, 1], c=uy, cmap=cm_bright, edgecolors='k')

plt.tight_layout()
plt.show()
