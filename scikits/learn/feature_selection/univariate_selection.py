"""
Univariate features selection.
"""

# Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay
# License: BSD 3 clause

import numpy as np
from scipy import stats

######################################################################
# Scoring functions
######################################################################

def f_classif(X, y):
    """
    Compute the Anova F-value for the provided sample

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        the set of regressors sthat will tested sequentially
    y : array of shape(n_samples)
        the data matrix

    Returns
    -------
    F : array of shape (m),
        the set of F values
    pval : array of shape(m),
        the set of p-values
    """
    X = np.asanyarray(X)
    args = [X[y==k] for k in np.unique(y)]
    return stats.f_oneway(*args)


def f_regression(X, y, center=True):
    """
    Quick linear model for testing the effect of a single regressor,
    sequentially for many regressors
    This is done in 3 steps:
    1. the regressor of interest and the data are orthogonalized
    wrt constant regressors
    2. the cross correlation between data and regressors is computed
    3. it is converted to an F score then to a p-value

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        the set of regressors sthat will tested sequentially
    y : array of shape(n_samples)
        the data matrix

    center : True, bool,
        If true, X and y are centered

    Returns
    -------
    F : array of shape (m),
        the set of F values
    pval : array of shape(m)
        the set of p-values
    """

    # orthogonalize everything wrt to confounds
    y = y.copy()
    X = X.copy()
    if center:
        y -= np.mean(y)
        X -= np.mean(X, 0)

    # compute the correlation
    X /= np.sqrt(np.sum(X**2,0))
    y /= np.sqrt(np.sum(y**2))
    corr = np.dot(y, X)

    # convert to p-value
    dof = y.size-2
    F = corr**2/(1-corr**2)*dof
    pv = stats.f.sf(F, 1, dof)
    return F, pv


######################################################################
# General class for filter univariate selection
######################################################################


class UnivariateFilter(object):

    def __init__(self, score_func):
        """
        Initialize the univariate feature selection.
        Func : function taking two arrays X and y, and returning an array.
        score_func returning both scores and pvalues
        """
        assert callable(score_func), ValueError(
                "The score function should be a callable, '%s' (type %s) "
                "was passed." % (score_func, type(score_func))
            )
        self.score_func = score_func

    def fit(self,X,y):
        """
        Evaluate the function
        """
        _scores = self.score_func(X, y)
        self._scores = _scores[0]
        self._pvalues = _scores[1]
        #self._rank = np.argsort(self._pvalues)
        return self

    def transform(self,X,**kwargs):
        """
        Transform a new matrix using the selected features
        """
        return X[:,self.support(**kwargs)]


######################################################################
# Specific filters
######################################################################

class SelectPercentile(UnivariateFilter):
    """
    Filter : Select the best percentile of the p_values
    """
    def support(self,percentile):
        assert percentile<=100, ValueError('percentile should be \
                            between 0 and 100 (%f given)' %(percentile))
        alpha = stats.scoreatpercentile(self._pvalues, percentile)
        return (self._pvalues <= alpha)

class SelectKBest(UnivariateFilter):
    """
    Filter : Select the k lowest p-values
    """
    def support(self,k):
        assert k<=len(self._pvalues), ValueError('cannot select %d features'
                                    ' among %d ' % (k, len(self._pvalues)))
        alpha = np.sort(self._pvalues)[k-1]
        return (self._pvalues <= alpha)

class SelectFpr(UnivariateFilter):
    """
    Filter : Select the pvalues below alpha
    """
    def support(self,alpha):
        return (self._pvalues < alpha)


class SelectFdr(UnivariateFilter):
    """
    Filter : Select the p-values corresponding to an estimated false
    discovery rate of alpha. This uses the Benjamini-Hochberg procedure
    """
    def support(self,alpha):
        sv = np.sort(self._pvalues)
        threshold = sv[sv < alpha*np.arange(len(self._pvalues))].max()
        return (self._pvalues < threshold)


class SelectFwe(UnivariateFilter):
    """
    Filter : Select the p-values corresponding to a corrected p-value of alpha
    """
    def support(self,alpha):
        return (self._pvalues < alpha/len(self._pvalues))



if __name__ == "__main__":
    import scikits.learn.datasets.samples_generator as sg
    from scikits.learn.svm import SVR, SVC

    X,y = sg.sparse_uncorrelated(50,100)
    univariate_filter = SelectKBest(f_regression)
    X_r = univariate_filter.fit(X, y).transform(X, k=5)
    sel = univariate_filter.support(k=5)
    clf = SVR(kernel='linear', C=1.)
    y_ = clf.fit(X_r, y).predict(X_r)
    print sel

    ### now change k
    X_r = univariate_filter.transform(X, k=2)
    y_ = clf.fit(X_r, y).predict(X)
    print sel


