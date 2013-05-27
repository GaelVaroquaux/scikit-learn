"""
Trivial tests of the tron solver
"""
import numpy as np

from sklearn.svm import _tron

def test_fmin_tron():
    func = lambda x: np.sum(x ** 2)
    def grad_hess(x):
        return (2*x, lambda x: 2*x)
    x0 = np.ones(10)
    x, res = _tron.fmin_tron(func, grad_hess, x0)
    np.testing.assert_almost_equal(x, np.zeros_like(x0))
