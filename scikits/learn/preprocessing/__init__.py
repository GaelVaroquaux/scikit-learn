""" Transformers to perform common preprocessing steps.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
# License: BSD

import numpy as np

from ..base import BaseEstimator


def _mean_and_std(X, axis=0, with_std=True):
    """Compute mean and std dev for centering, scaling

    Zero valued std components are reseted to 1.0 to avoid NaNs when scaling.
    """
    Xr = np.rollaxis(X, axis)
    mean_ = Xr.mean(axis=0)

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.0] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


def scale(X, axis=0, with_std=True, copy=True):
    """Method to standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.
    """
    mean_, std_ = _mean_and_std(X, axis, with_std)
    if copy:
        X = X.copy()
    Xr = np.rollaxis(X, axis)
    Xr -= mean_
    if with_std:
        Xr /= std_
    return X


class Scaler(BaseEstimator):
    """Object to standardize a dataset

    It centers the dataset and optionaly scales to fix the variance to 1 for
    each feature
    """

    def __init__(self, with_std=True):
        self.with_std = with_std

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        self.mean_, self.std_ = _mean_and_std(X, axis=0,
                                              with_std=self.with_std)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()
        # We are taking a view of the X array and modifying it
        X -= self.mean_
        if self.with_std:
            X /= self.std_
        return X


class Normalizer(BaseEstimator):
    """Normalize vectors such that they sum to 1"""

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()
        norms = X.sum(axis=1)[:, np.newaxis]
        norms[norms == 0.0] = 1.0
        X /= norms

        return X


class LengthNormalizer(BaseEstimator):
    """Normalize vectors to unit vectors"""

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()

        norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
        norms[norms == 0.0] = 1.0
        X /= norms

        return X


class Binarizer(BaseEstimator):
    """Binarize data according to a threshold"""

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None, **params):
        self._set_params(**params)
        return self

    def transform(self, X, y=None, copy=True):
        if copy:
            X = X.copy()

        cond = X > self.threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0

        return X


class LabelBinarizer(BaseEstimator):
    """Binarize labels in a one-vs-all fashion, a.k.a 1-of-K coding scheme"""

    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        self.classes = np.unique(y)
        return self

    def transform(self, y):
        """Transform multi-class labels to binary labels

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes]
        """
        if len(self.classes) == 2:
            Y = np.zeros((len(y), 1))
            Y[y == self.classes[1], 0] = 1
            return Y

        elif len(self.classes) >= 2:
            Y = np.zeros((len(y), len(self.classes)))
            for i, k in enumerate(self.classes):
                Y[y == k, i] = 1
            return Y

        else:
            raise ValueError

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array of shape [n_samples, n_classes]
            Target values

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        if len(Y.shape) == 1 or Y.shape[1] == 1:
            y = np.array(Y.ravel() > 0, dtype=int)
        else:
            y = Y.argmax(axis=1)
        return self.classes[y]
