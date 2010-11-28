"""
Generate samples of synthetic data sets.
"""

# Author: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel
# License: BSD 3 clause

import numpy as np
import numpy.random as nr


def test_dataset_classif(n_samples=100, n_features=100, param=[1,1],
                         n_informative=0, k=0, seed=None):
    """Generate an snp matrix

    Parameters
    ----------
    n_samples : 100, int,
        the number of observations

    n_features : 100, int,
        the number of features for each observation

    param : [1, 1], list,
        parameter of a dirichlet density
        that is used to generate multinomial densities
        from which the n_features will be samples

    n_informative: 0, int
        number of informative features

    k : 0, int
        deprecated: use n_informative instead

    seed : None, int or np.random.RandomState
        if seed is an instance of np.random.RandomState,
        it is used to initialize the random generator

    Returns
    -------
    x : array of shape(n_samples, n_features),
        the design matrix

    y : array of shape (n_samples),
        the subject labels

    """
    if k > 0 and n_informative == 0:
        n_informative = k

    if n_informative > n_features:
        raise ValueError('cannot have %d informative features and'
                         ' %d features' % (n_informative, n_features))

    if isinstance(seed, np.random.RandomState):
        random = seed
    elif seed is not None:
        random = np.random.RandomState(seed)
    else:
        random = np.random

    x = random.randn(n_samples, n_features)
    y = np.zeros(n_samples)
    param = np.ravel(np.array(param)).astype(np.float)
    for n in range(n_samples):
        y[n] = np.nonzero(random.multinomial(1, param / param.sum()))[0]
    x[:, :k] += 3 * y[:, np.newaxis]
    return x, y


def test_dataset_reg(n_samples=100, n_features=100, n_informative=0, k=0,
                     seed=None):
    """Generate an snp matrix

    Parameters
    ----------
    n_samples : 100, int
        the number of subjects

    n_features : 100, int
        the number of features

    n_informative: 0, int
        number of informative features

    k : 0, int
        deprecated: use n_informative instead

    seed : None, int or np.random.RandomState
        if seed is an instance of np.random.RandomState,
        it is used to initialize the random generator

    Returns
    -------
    x : array of shape(n_samples, n_features),
        the design matrix

    y : array of shape (n_samples),
        the subject data
    """
    if k > 0 and n_informative == 0:
        n_informative = k

    if n_informative > n_features:
        raise ValueError('cannot have %d informative features and'
                         ' %d features' % (n_informative, n_features))

    if isinstance(seed, np.random.RandomState):
        random = seed
    elif seed is not None:
        random = np.random.RandomState(seed)
    else:
        random = np.random

    x = random.randn(n_samples, n_features)
    y = random.randn(n_samples)
    x[:, :k] += y[:, np.newaxis]
    return x, y


def sparse_uncorrelated(n_samples=100, n_features=10):
    """Function creating simulated data with sparse uncorrelated design

    cf.Celeux et al. 2009,  Bayesian regularization in regression)

    X = NR.normal(0, 1)
    Y = NR.normal(X[:, 0] + 2 * X[:, 1] - 2 * X[:, 2] - 1.5 * X[:, 3])
    The number of features is at least 10.

    Parameters
    ----------
    n_samples : int
        number of samples (default is 100).
    n_features : int
        number of features (default is 10).

    Returns
    -------
    X : numpy array of shape (n_samples, n_features) for input samples
    y : numpy array of shape (n_samples) for labels
    """
    X = nr.normal(loc=0, scale=1, size=(n_samples, n_features))
    y = nr.normal(loc=X[:, 0] + 2 * X[:, 1] - 2 * X[:,2] - 1.5 * X[:, 3],
                  scale = np.ones(n_samples))
    return X, y


def friedman(n_samples=100, n_features=10, noise_std=1):
    """Function creating simulated data with non linearities

    cf. Friedman 1993

    X = np.random.normal(0, 1)

    y = 10 * sin(X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
            + 10 * X[:, 3] + 5 * X[:, 4]

    The number of features is at least 5.

    Parameters
    ----------
    n_samples : int
        number of samples (default is 100).

    n_features : int
        number of features (default is 10).

    noise_std : float
        std of the noise, which is added as noise_std*NR.normal(0,1)

    Returns
    -------
    X : numpy array of shape (n_samples, n_features) for input samples
    y : numpy array of shape (n_samples,) for labels
    """
    X = nr.normal(loc=0, scale=1, size=(n_samples, n_features))
    y = 10 * np.sin(X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 \
            + 10 * X[:, 3] + 5 * X[:, 4]
    y += noise_std * nr.normal(loc=0, scale=1, size=n_samples)
    return X, y

