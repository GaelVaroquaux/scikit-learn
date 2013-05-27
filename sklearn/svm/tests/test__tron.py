"""
Trivial tests of the tron solver
"""
import numpy as np

from sklearn.svm import _tron

def test_fmin_tron():
    func = lambda x: x ** 2
    def grad_hess(x):
        return (x, lambda x: x)
    x0 = np.ones(10)
    res = _tron.fmin_tron(func, grad_hess, x0)
    x = res.x
    np.testing.assert_almost_equal(x, np.zeros_like(x0))

