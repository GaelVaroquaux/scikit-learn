# Authors: Peter Prettenhofer <peter.prettenhofer@gmail.com> (main author)
#          Mathieu Blondel (partial_fit support)
#
# License: BSD Style.
"""Classification and regression using Stochastic Gradient Descent (SGD)."""

import numpy as np
import scipy.sparse as sp

from abc import ABCMeta, abstractmethod
import warnings

from ..externals.joblib import Parallel, delayed

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..feature_selection.selector_mixin import SelectorMixin
from ..utils import array2d, atleast2d_or_csr, check_arrays, safe_asarray
from ..utils.extmath import safe_sparse_dot
from ..utils import deprecated

from .sgd_fast import plain_sgd as plain_sgd
from ..utils.seq_dataset import ArrayDataset, CSRDataset
from .sgd_fast import Hinge
from .sgd_fast import Log
from .sgd_fast import ModifiedHuber
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive


class BaseSGD(BaseEstimator):
    """Base class for SGD classification and regression."""

    __metaclass__ = ABCMeta

    def __init__(self, loss, penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, seed=0, learning_rate="optimal",
                 eta0=0.0, power_t=0.5, warm_start=False):
        self.loss = str(loss)
        self.penalty = str(penalty).lower()
        self.epsilon = float(epsilon)
        self._set_loss_function(self.loss)
        self._set_penalty_type(self.penalty)

        self.alpha = float(alpha)
        if self.alpha < 0.0:
            raise ValueError("alpha must be greater than zero")
        self.rho = float(rho)
        if self.rho < 0.0 or self.rho > 1.0:
            raise ValueError("rho must be in [0, 1]")
        self.fit_intercept = bool(fit_intercept)
        self.n_iter = int(n_iter)
        if self.n_iter <= 0:
            raise ValueError("n_iter must be greater than zero")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.verbose = int(verbose)

        self.learning_rate = str(learning_rate)
        self._set_learning_rate(self.learning_rate)
        self.eta0 = float(eta0)
        self.power_t = float(power_t)
        if self.learning_rate != "optimal":
            if eta0 <= 0.0:
                raise ValueError("eta0 must be greater than 0.0")
        self.coef_ = None
        self.warm_start = warm_start

        self._init_t()

    @abstractmethod
    def fit(self, X, y):
        """Fit model."""

    @abstractmethod
    def predict(self, X):
        """Predict using model."""

    def _init_t(self):
        self.t_ = 1.0
        if self.learning_rate == "optimal":
            typw = np.sqrt(1.0 / np.sqrt(self.alpha))
            # computing eta0, the initial learning rate
            eta0 = typw / max(1.0, self.loss_function.dloss(-typw, 1.0))
            # initialize t such that eta at first example equals eta0
            self.t_ = 1.0 / (eta0 * self.alpha)

    def _set_learning_rate(self, learning_rate):
        learning_rate_codes = {"constant": 1, "optimal": 2, "invscaling": 3}
        try:
            self.learning_rate_code = learning_rate_codes[learning_rate]
        except KeyError:
            raise ValueError("learning rate %s"
                             "is not supported. " % learning_rate)

    def _set_penalty_type(self, penalty):
        penalty_types = {"none": 0, "l2": 2, "l1": 1, "elasticnet": 3}
        try:
            self.penalty_type = penalty_types[penalty]
        except KeyError:
            raise ValueError("Penalty %s is not supported. " % penalty)

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight

    def _set_coef(self, coef_):
        """Make sure that coef_ is fortran-style and 2d.

        Fortran-style memory layout is needed to ensure that computing
        the dot product between input ``X`` and ``coef_`` does not trigger
        a memory copy.
        """
        self.coef_ = np.asfortranarray(array2d(coef_))

    def _allocate_parameter_mem(self, n_classes, n_features, coef_init=None,
                                intercept_init=None):
        """Allocate mem for parameters; initialize if provided."""
        if n_classes > 2:
            # allocate coef_ for multi-class
            if coef_init is not None:
                coef_init = np.asarray(coef_init, order="C")
                if coef_init.shape != (n_classes, n_features):
                    raise ValueError("Provided coef_ does not match dataset. ")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros((n_classes, n_features),
                                      dtype=np.float64, order="C")

            # allocate intercept_ for multi-class
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, order="C")
                if intercept_init.shape != (n_classes, ):
                    raise ValueError("Provided intercept_init " \
                                     "does not match dataset.")
                self.intercept_ = intercept_init
            else:
                self.intercept_ = np.zeros(n_classes, dtype=np.float64,
                                           order="C")
        else:
            # allocate coef_ for binary problem
            if coef_init is not None:
                coef_init = np.asarray(coef_init, dtype=np.float64,
                                       order="C")
                coef_init = coef_init.ravel()
                if coef_init.shape != (n_features,):
                    raise ValueError("Provided coef_init does not " \
                                     "match dataset.")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")

            # allocate intercept_ for binary problem
            if intercept_init is not None:
                intercept_init = np.asarray(intercept_init, dtype=np.float64)
                if intercept_init.shape != (1,) and intercept_init.shape != ():
                    raise ValueError("Provided intercept_init " \
                                     "does not match dataset.")
                self.intercept_ = intercept_init.reshape(1,)
            else:
                self.intercept_ = np.zeros(1, dtype=np.float64, order="C")


def _check_fit_data(X, y):
    n_samples, _ = X.shape
    if n_samples != y.shape[0]:
        raise ValueError("Shapes of X and y do not match.")


def _make_dataset(X, y_i, sample_weight):
    """Returns Dataset object + intercept_decay"""
    if sp.issparse(X):
        dataset = CSRDataset(X.data, X.indptr, X.indices, y_i, sample_weight)
        intercept_decay = 0.01
    else:
        dataset = ArrayDataset(X, y_i, sample_weight)
        intercept_decay = 1.0
    return dataset, intercept_decay


def _tocsr(X):
    """Convert X to CSR matrix, preventing a copy if possible"""
    if sp.isspmatrix_csr(X) and X.dtype == np.float64:
        return X
    else:
        return sp.csr_matrix(X, dtype=np.float64)


class SGDClassifier(BaseSGD, ClassifierMixin, SelectorMixin):
    """Linear model fitted by minimizing a regularized empirical loss with SGD.

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Parameters
    ----------
    loss : str, 'hinge' or 'log' or 'modified_huber'
        The loss function to be used. Defaults to 'hinge'. The hinge loss is
        a margin loss used by standard linear SVM models. The 'log' loss is
        the loss of logistic regression models and can be used for
        probability estimation in binary classifiers. 'modified_huber'
        is another smooth loss that brings tolerance to outliers.

    penalty : str, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' migh bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001

    rho : float
        The Elastic Net mixing parameter, with 0 < rho <= 1.
        Defaults to 0.85.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter: int, optional
        The number of passes over the training data (aka epochs).
        Defaults to 5.

    shuffle: bool, optional
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.

    seed: int, optional
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose: integer, optional
        The verbosity level

    n_jobs: integer, optional
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation. -1 means 'all CPUs'. Defaults
        to 1.

    learning_rate : string, optional
        The learning rate:
        constant: eta = eta0
        optimal: eta = 1.0/(t+t0) [default]
        invscaling: eta = eta0 / pow(t, power_t)

    eta0 : double
        The initial learning rate [default 0.01].

    power_t : double
        The exponent for inverse scaling learning rate [default 0.25].

    class_weight : dict, {class_label : weight} or "auto" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "auto" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    `coef_` : array, shape = [1, n_features] if n_classes == 2 else [n_classes,
    n_features]
        Weights assigned to the features.

    `intercept_` : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> Y = np.array([1, 1, 2, 2])
    >>> clf = linear_model.SGDClassifier()
    >>> clf.fit(X, Y)
    ... #doctest: +NORMALIZE_WHITESPACE
    SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
            fit_intercept=True, learning_rate='optimal', loss='hinge',
            n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, rho=0.85, seed=0,
            shuffle=False, verbose=0, warm_start=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    LinearSVC, LogisticRegression, Perceptron

    """
    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, n_jobs=1, seed=0,
                 learning_rate="optimal", eta0=0.0, power_t=0.5,
                 class_weight=None, warm_start=False):
        super(SGDClassifier, self).__init__(loss=loss, penalty=penalty,
                                            alpha=alpha, rho=rho,
                                            fit_intercept=fit_intercept,
                                            n_iter=n_iter, shuffle=shuffle,
                                            verbose=verbose, epsilon=epsilon,
                                            seed=seed,
                                            learning_rate=learning_rate,
                                            eta0=eta0, power_t=power_t,
                                            warm_start=warm_start)
        self.class_weight = class_weight
        self.classes_ = None
        self.n_jobs = int(n_jobs)

    @property
    @deprecated("to be removed in v0.13; use ``classes_`` instead.")
    def classes(self):
        return self.classes_

    def _set_loss_function(self, loss):
        """Set concrete LossFunction."""
        loss_functions = {
            "hinge": Hinge(1.0),
            "perceptron": Hinge(0.0),
            "log": Log(),
            "modified_huber": ModifiedHuber(),
            "squared_loss": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon),
        }
        try:
            self.loss_function = loss_functions[loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % loss)

    def _set_class_weight(self, class_weight, classes, y):
        """Estimate class weights for unbalanced datasets."""
        if class_weight is None or len(class_weight) == 0:
            # uniform class weights
            weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        elif class_weight == 'auto':
            # proportional to the number of samples in the class
            weight = np.array([1.0 / np.sum(y == i) for i in classes],
                              dtype=np.float64, order='C')
            weight *= classes.shape[0] / np.sum(weight)
        else:
            # user-defined dictionary
            weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
            if not isinstance(class_weight, dict):
                raise ValueError("class_weight must be dict, 'auto', or None,"
                                 " got: %r" % class_weight)
            for c in class_weight:
                i = np.searchsorted(classes, c)
                if classes[i] != c:
                    raise ValueError("Class label %d not present." % c)
                else:
                    weight[i] = class_weight[c]

        self._expanded_class_weight = weight

    def _partial_fit(self, X, y, n_iter, classes=None, sample_weight=None,
                     coef_init=None, intercept_init=None):
        X = safe_asarray(X, dtype=np.float64, order="C")
        y = np.asarray(y)

        n_samples, n_features = X.shape
        _check_fit_data(X, y)

        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call "
                             "to partial_fit.")
        elif classes is not None and self.classes_ is not None:
            if not np.all(self.classes_ == np.unique(classes)):
                raise ValueError("`classes` is not the same as on last call "
                                 "to partial_fit.")
        elif classes is not None:
            self.classes_ = classes

        n_classes = self.classes_.shape[0]

        # Allocate datastructures from input arguments
        self._set_class_weight(self.class_weight, self.classes_, y)
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        if self.coef_ is None:
            self._allocate_parameter_mem(n_classes, n_features,
                                         coef_init, intercept_init)

        # delegate to concrete training procedure
        if n_classes > 2:
            self._fit_multiclass(X, y, sample_weight, n_iter)
        elif n_classes == 2:
            self._fit_binary(X, y, sample_weight, n_iter)
        else:
            raise ValueError("The number of class labels must be "
                             "greater than one.")

        self.t_ += n_iter * n_samples

        return self

    def partial_fit(self, X, y, classes=None,
                    class_weight=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Subset of the training data

        y : numpy array of shape [n_samples]
            Subset of the target values

        classes : array, shape = [n_classes]
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        if class_weight is not None:
            warnings.warn("Using 'class_weight' as a parameter to the 'fit'"
                          "method is deprecated and will be removed in 0.13. "
                          "Set it on initialization instead.",
                          DeprecationWarning, stacklevel=2)
            self.class_weight = class_weight
        return self._partial_fit(X, y, n_iter=1, classes=classes,
                                 sample_weight=sample_weight)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            class_weight=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_classes,n_features]
            The initial coeffients to warm-start the optimization.

        intercept_init : array, shape = [n_classes]
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        if class_weight is not None:
            warnings.warn("Using 'class_weight' as a parameter to the 'fit'"
                          "method is deprecated and will be removed in 0.13. "
                          "Set it on initialization instead.",
                          DeprecationWarning, stacklevel=2)

            self.class_weight = class_weight

        X = safe_asarray(X, dtype=np.float64, order="C")
        # labels can be encoded as float, int, or string literals
        y = np.asarray(y)

        n_samples, n_features = X.shape
        _check_fit_data(X, y)

        # np.unique sorts in asc order; largest class id is positive class
        classes = np.unique(y)

        if self.warm_start and self.coef_ is not None:
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None

        # Need to re-initialize in case of multiple call to fit.
        self._init_t()

        self._partial_fit(X, y, self.n_iter, classes,
                          sample_weight, coef_init, intercept_init)

        # fitting is over, we can now transform coef_ to fortran order
        # for faster predictions
        self._set_coef(self.coef_)

        return self

    def decision_function(self, X):
        """Predict signed 'distance' to the hyperplane (aka confidence score)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples,n_classes]
            The signed 'distances' to the hyperplane(s).
        """
        X = atleast2d_or_csr(X)
        scores = safe_sparse_dot(X, self.coef_.T) + self.intercept_
        if self.classes_.shape[0] == 2:
            return np.ravel(scores)
        else:
            return scores

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        scores = self.decision_function(X)
        if self.classes_.shape[0] == 2:
            indices = np.array(scores > 0, dtype=np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[np.ravel(indices)]

    def predict_proba(self, X):
        """Predict class membership probability

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples,
        n_classes]
            Contains the membership probabilities of the positive class.

        References
        ----------

        The justification for the formula in the loss="modified_huber"
        case is in the appendix B in:
        http://jmlr.csail.mit.edu/papers/volume2/zhang02c/zhang02c.pdf
        """
        if len(self.classes_) != 2:
            raise NotImplementedError("predict_(log_)proba only supported"
                                      " for binary classification")

        proba = np.ones((len(X), 2), dtype=np.float64)
        if self.loss == "log":
            proba[:, 1] = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        elif self.loss == "modified_huber":
            proba[:, 1] = np.minimum(1, np.maximum(-1,
                                                   self.decision_function(X)))
            proba[:, 1] += 1
            proba[:, 1] /= 2
        else:
            raise NotImplementedError("predict_(log_)proba only supported when"
                                      " loss='log' or loss='modified_huber' "
                                      "(%s given)" % self.loss)
        proba[:, 0] -= proba[:, 1]
        return proba

    def _fit_binary(self, X, y, sample_weight, n_iter):
        if sp.issparse(X):
            X = _tocsr(X)

        coef, intercept = fit_binary(self, 1, X, y, n_iter,
                                     self._expanded_class_weight[1],
                                     self._expanded_class_weight[0],
                                     sample_weight)

        # need to be 2d
        self.coef_ = coef.reshape(1, -1)
        # intercept is a float, need to convert it to an array of length 1
        self.intercept_ = np.atleast_1d(intercept)

    def _fit_multiclass(self, X, y, sample_weight, n_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OVA: One Versus All.
        """
        if sp.issparse(X):
            X = _tocsr(X)

        # Use joblib to fit OvA in parallel
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fit_binary)(self, i, X, y, n_iter,
                                self._expanded_class_weight[i], 1.,
                                sample_weight)
            for i in xrange(len(self.classes_)))

        for i, (coef, intercept) in enumerate(result):
            self.coef_[i] = coef
            self.intercept_[i] = intercept


def _prepare_fit_binary(est, y, i):
    """Common initialization for _fit_binary_{dense,sparse}.

    Returns y, coef, intercept.
    """
    y_i = np.ones(y.shape, dtype=np.float64, order="C")
    y_i[y != est.classes_[i]] = -1.0

    if len(est.classes_) == 2:
        coef = est.coef_.ravel()
        intercept = est.intercept_[0]
    else:
        coef = est.coef_[i]
        intercept = est.intercept_[i]

    return y_i, coef, intercept


def fit_binary(est, i, X, y, n_iter, pos_weight, neg_weight,
               sample_weight):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.
    """
    y_i, coef, intercept = _prepare_fit_binary(est, y, i)
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]
    dataset, intercept_decay = _make_dataset(X, y_i, sample_weight)

    return plain_sgd(coef, intercept, est.loss_function,
                     est.penalty_type, est.alpha, est.rho,
                     dataset, n_iter, est.fit_intercept,
                     est.verbose, est.shuffle, est.seed,
                     pos_weight, neg_weight,
                     est.learning_rate_code, est.eta0,
                     est.power_t, est.t_, intercept_decay)


class SGDRegressor(BaseSGD, RegressorMixin, SelectorMixin):
    """Linear model fitted by minimizing a regularized empirical loss with SGD

    SGD stands for Stochastic Gradient Descent: the gradient of the loss is
    estimated each sample at a time and the model is updated along the way with
    a decreasing strength schedule (aka learning rate).

    The regularizer is a penalty added to the loss function that shrinks model
    parameters towards the zero vector using either the squared euclidean norm
    L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
    parameter update crosses the 0.0 value because of the regularizer, the
    update is truncated to 0.0 to allow for learning sparse models and achieve
    online feature selection.

    This implementation works with data represented as dense numpy arrays of
    floating point values for the features.

    Parameters
    ----------
    loss : str, 'squared_loss' or 'huber'
        The loss function to be used. Defaults to 'squared_loss' which refers
        to the ordinary least squares fit. 'huber' is an epsilon insensitive
        loss function for robust regression.

    penalty : str, 'l2' or 'l1' or 'elasticnet'
        The penalty (aka regularization term) to be used. Defaults to 'l2'
        which is the standard regularizer for linear SVM models. 'l1' and
        'elasticnet' migh bring sparsity to the model (feature selection)
        not achievable with 'l2'.

    alpha : float
        Constant that multiplies the regularization term. Defaults to 0.0001

    rho : float
        The Elastic Net mixing parameter, with 0 < rho <= 1.
        Defaults to 0.85.

    fit_intercept: bool
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered. Defaults to True.

    n_iter: int, optional
        The number of passes over the training data (aka epochs).
        Defaults to 5.

    shuffle: bool, optional
        Whether or not the training data should be shuffled after each epoch.
        Defaults to False.

    seed: int, optional
        The seed of the pseudo random number generator to use when
        shuffling the data.

    verbose: integer, optional
        The verbosity level.

    epsilon: float
        Epsilon in the epsilon-insensitive huber loss function;
        only if `loss=='huber'`.

    learning_rate : string, optional
        The learning rate:
        constant: eta = eta0
        optimal: eta = 1.0/(t+t0)
        invscaling: eta = eta0 / pow(t, power_t) [default]

    eta0 : double, optional
        The initial learning rate [default 0.01].

    power_t : double, optional
        The exponent for inverse scaling learning rate [default 0.25].

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    `coef_` : array, shape = [n_features]
        Weights asigned to the features.

    `intercept_` : array, shape = [1]
        The intercept term.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import linear_model
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = linear_model.SGDRegressor()
    >>> clf.fit(X, y)
    SGDRegressor(alpha=0.0001, epsilon=0.1, eta0=0.01, fit_intercept=True,
           learning_rate='invscaling', loss='squared_loss', n_iter=5, p=None,
           penalty='l2', power_t=0.25, rho=0.85, seed=0, shuffle=False,
           verbose=0, warm_start=False)

    See also
    --------
    Ridge, ElasticNet, Lasso, SVR

    """
    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
            rho=0.85, fit_intercept=True, n_iter=5, shuffle=False, verbose=0,
            epsilon=0.1, p=None, seed=0, learning_rate="invscaling", eta0=0.01,
            power_t=0.25, warm_start=False):

        if p is not None:
            warnings.warn("Using 'p' is deprecated and will be removed in "
                          "scikit-learn 0.14, use epsilon instead.",
                           DeprecationWarning, stacklevel=2)
            self.p = float(p)
            epsilon = p

        super(SGDRegressor, self).__init__(loss=loss, penalty=penalty,
                                           alpha=alpha, rho=rho,
                                           fit_intercept=fit_intercept,
                                           n_iter=n_iter, shuffle=shuffle,
                                           verbose=verbose, epsilon=epsilon,
                                           seed=seed,
                                           learning_rate=learning_rate,
                                           eta0=eta0, power_t=power_t,
                                           warm_start=False)

    def _set_loss_function(self, loss):
        """Get concrete LossFunction"""
        loss_functions = {
            "squared_loss": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon),
        }
        try:
            self.loss_function = loss_functions[loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % loss)

    def _partial_fit(self, X, y, n_iter, sample_weight=None,
                     coef_init=None, intercept_init=None):
        X, y = check_arrays(X, y, sparse_format="csr", copy=False,
                            check_ccontiguous=True, dtype=np.float64)

        n_samples, n_features = X.shape
        _check_fit_data(X, y)

        # Allocate datastructures from input arguments
        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        if self.coef_ is None:
            self._allocate_parameter_mem(1, n_features,
                                         coef_init, intercept_init)

        self._fit_regressor(X, y, sample_weight, n_iter)

        self.t_ += n_iter * n_samples

        return self

    def partial_fit(self, X, y, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Subset of training data

        y : numpy array of shape [n_samples]
            Subset of target values

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """
        return self._partial_fit(X, y, n_iter=1, sample_weight=sample_weight)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_features]
            The initial coeffients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : returns an instance of self.
        """
        if self.warm_start and self.coef_ is not None:
            if coef_init is None:
                coef_init = self.coef_
            if intercept_init is None:
                intercept_init = self.intercept_
        else:
            self.coef_ = None
            self.intercept_ = None

        # Need to re-initialize in case of multiple call to fit.
        self._init_t()

        return self._partial_fit(X, y, self.n_iter, sample_weight,
                                 coef_init, intercept_init)

    def decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)
        scores = safe_sparse_dot(X, self.coef_) + self.intercept_
        return scores.ravel()

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
           Predicted target values per element in X.
        """
        return self.decision_function(X)

    def _fit_regressor(self, X, y, sample_weight, n_iter):
        dataset, intercept_decay = _make_dataset(X, y, sample_weight)

        self.coef_, intercept = plain_sgd(self.coef_,
                                          self.intercept_[0],
                                          self.loss_function,
                                          self.penalty_type,
                                          self.alpha, self.rho,
                                          dataset,
                                          n_iter,
                                          int(self.fit_intercept),
                                          int(self.verbose),
                                          int(self.shuffle),
                                          self.seed,
                                          1.0, 1.0,
                                          self.learning_rate_code,
                                          self.eta0, self.power_t, self.t_,
                                          intercept_decay)

        self.intercept_ = np.atleast_1d(intercept)
