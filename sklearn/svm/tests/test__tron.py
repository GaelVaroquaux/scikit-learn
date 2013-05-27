"""
Trivial tests of the tron solver
"""
import numpy as np

from sklearn.svm import _tron

def test_fmin_tron():
    func = lambda x: x ** 2
    grad_hess = lambda x: (x, np.identity(x.shape[0]))
    x0 = np.ones(10)
    res = _tron.fmin_tron(func, grad_hess, x0)
    x = res.x
    np.testing.assert_almost_equal(x, np.zeros_like(x0))

