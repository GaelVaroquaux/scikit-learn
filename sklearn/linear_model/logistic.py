import numpy as np
from scipy import optimize

from .base import LinearClassifierMixin, SparseCoefMixin
from ..feature_selection.from_model import _LearntSelectorMixin
from ..svm.base import BaseLibLinear
from ..svm._tron import fmin_tron


class LogisticRegression(BaseLibLinear, LinearClassifierMixin,
                         _LearntSelectorMixin, SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses a one-vs.-all (OvA)
    scheme, rather than the "true" multinomial LR.

    This class implements L1 and L2 regularized logistic regression using the
    `liblinear` library. It can handle both dense and sparse input. Use
    C-ordered arrays or CSR matrices containing 64-bit floats for optimal
    performance; any other input format will be converted (and copied).

    Parameters
    ----------
    penalty : string, 'l1' or 'l2'
        Used to specify the norm used in the penalization.

    dual : boolean
        Dual or primal formulation. Dual formulation is only
        implemented for l2 penalty. Prefer dual=False when
        n_samples > n_features.

    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.

    intercept_scaling : float, default: 1
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    tol: float, optional
        Tolerance for stopping criteria.

    Attributes
    ----------
    `coef_` : array, shape = [n_classes-1, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes-1]
        Intercept (a.k.a. bias) added to the decision function.
        It is available only when parameter intercept is set to True.

    random_state: int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    See also
    --------
    LinearSVC

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    References:

    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):

        super(LogisticRegression, self).__init__(
            penalty=penalty, dual=dual, loss='lr', tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        return self._predict_proba_lr(X)

    def predict_log_proba(self, X):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


###############################################################################
# Solver based directly on the TRON optimizer


def _phi(t, copy=True):
    # helper function: return 1. / (1 + np.exp(-t))
    if copy:
        t = np.copy(t)
    t *= -1.
    t = np.exp(t, t)
    t += 1
    t = np.reciprocal(t, t)
    return t


def _logistic_loss(w, X, y, alpha):
    # loss function to be optimized, it's the logistic loss
    z = X.dot(w)
    yz = y * z
    idx = yz > 0
    out = np.empty(yz.shape, yz.dtype)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum() + .5 * alpha * w.dot(w)
    print 'Logistic value: %f (norm of input %f)' % (out, np.sum(w**2))
    return out


def _logistic_grad_hess(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    z = _phi(y * z, copy=False)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w

    # The mat-vec product of the Hessian
    d = z * (1 - z)
    # Precompute as much as possible
    d = np.sqrt(d, d)
    dX = d[:, np.newaxis] * X
    def Hs(s):
        return dX.T.dot(dX.dot(s)) + alpha * s
    return grad, Hs


def _logistic_grad(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    z = _phi(y * z, copy=False)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    return grad


def _logistic_loss_and_grad(w, X, y, alpha):
    # gradient of the logistic loss
    z = X.dot(w)
    yz = y * z
    idx = yz > 0
    out = np.empty(yz.shape, yz.dtype)
    out[idx] = np.log(1 + np.exp(-yz[idx]))
    out[~idx] = (-yz[~idx] + np.log(1 + np.exp(yz[~idx])))
    out = out.sum() + .5 * alpha * w.dot(w)

    z = _phi(yz, copy=False)
    z0 = (z - 1) * y
    grad = X.T.dot(z0) + alpha * w
    return out, grad

from .base import center_data

def logistic_regression(X, y, C=1., w0=None, max_iter=15, gtol=1e-3,
                        tol=1e-12, solver='lbfgs', verbose=0):
    # Convert y to [-1, 1] values
    assert len(np.unique(y)) == 2
    y = y - y.min()
    y = np.sign(y)
    y = y.astype(np.float)
    X, _, _, _, _ = center_data(X, y, fit_intercept=True,
                                normalize=False)
    if w0 is None:
        n_samples, n_features = X.shape
        w0 = np.ones(n_features)
        # We don't want sum(w0) too big, because it leads to exploring
        # parts of the logistic loss where it is flat, and thus hard to
        # optimize
        w0 /= n_features
    if solver == 'lbfgs':
        #out = optimize.fmin_l_bfgs_b(_logistic_loss_and_grad, w0,
        #                             args=(X, y, 1./C), iprint=verbose > 0)
        out = optimize.fmin_l_bfgs_b(_logistic_loss, w0,
                                     fprime=_logistic_grad,
                                     args=(X, y, 1./C), iprint=verbose > 0)
        return out[0]
    else:
        # Bypass the checks
        from ..svm._tron_fast import _fmin_tron
        # Use a very small starting point, to be in the trust region
        w0 /= n_features
        w, res = _fmin_tron(_logistic_loss, _logistic_grad_hess, w0,
                        args=(X, y, 1./C), max_iter=max_iter, gtol=gtol,
                        tol=tol)
    return w



