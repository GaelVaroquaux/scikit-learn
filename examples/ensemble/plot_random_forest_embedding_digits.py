"""
=============================================================================
Random forest embedding on handwritten digits
=============================================================================

"""

# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Gael Varoquaux
# License: BSD

print __doc__
from time import time

import numpy as np
import pylab as pl
from matplotlib import offsetbox
from sklearn import ensemble, datasets, decomposition

digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    pl.figure()
    ax = pl.subplot(111)
    for i in range(X.shape[0]):
        pl.text(X[i, 0], X[i, 1], str(digits.target[i]),
                color=pl.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=pl.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    pl.xticks([]), pl.yticks([])
    if title is not None:
        pl.title(title)


#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print "Computing Random Forest embedding"
# use RandomForestEmbedding to transform data
hasher = ensemble.RandomForestEmbedding(n_estimators=200, random_state=0,
                                        max_depth=3)
t0 = time()
X_transformed = hasher.fit_transform(X)
pca = decomposition.RandomizedPCA(n_components=2)
X_reduced = pca.fit_transform(X_transformed)

plot_embedding(X_reduced,
    "Random forest embedding of the digits (time %.2fs)" %
    (time() - t0))

pl.show()
