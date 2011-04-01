import numpy as np
from .. import nmf
from nose.tools import ok_, assert_true, assert_false, raises

@raises(ValueError)
def test_initialize_nn_input():
    """
    Test _initialize_nmf_ behaviour on negative input
    """
    nmf._initialize_nmf_(-np.ones((2,2)), 2)

def test_initialize_nn_output():
    """
    Test that _initialize_nmf_ does not suggest negative values anywhere.
    """

    data = np.abs(np.random.randn(10,10))
    W, H = nmf._initialize_nmf_(data, 10)
    assert_false((W < 0).any() or (H < 0).any())

def test_initialize_close():
    """
    Test that _initialize_nmf_ error is
    less than the standard deviation 
    of the entries in the matrix
    """
    A = np.abs(np.random.randn(10,10))
    W, H = nmf._initialize_nmf_(A, 10)
    error = np.linalg.norm(np.dot(W, H) - A)
    sdev = np.linalg.norm(A - A.mean())
    assert_true(error <= sdev)

@raises(ValueError)
def test_fit_nn_input():
    """
    Test model fit behaviour on negative input
    """
    A = -np.ones((2,2)), 2
    m = nmf.NMF(2, initial=None)
    m.fit(A)

def test_fit_nn_output():
    """
    Test that the model does not use negative values anywhere
    """
    A = np.c_[5 * np.ones(5) - xrange(1, 6),
              5 * np.ones(5) + xrange(1, 6)]
    model = nmf.NMF(2, initial=None)
    model.fit(A)
    assert_false((model.components_ < 0).any() or
                 (model.data_ < 0).any())

def test_fit_nn_close():
    """
    Test that the fit is "close enough"
    """
    assert nmf.NMF(5).fit(np.abs(
      np.random.randn(6, 5))).reconstruction_err_ < 0.01

@raises(ValueError)
def test_nls_nn_input():
    """
    Test NLS behaviour on negative input
    """
    A = np.ones((2,2))
    nmf._nls_subproblem_(A, A, -A, 0.001, 20)

def test_nls_nn_output():
    """
    Test NLS doesn't return negative input.
    """
    A = np.atleast_2d(range(1,5))
    Ap, _, _ = nmf._nls_subproblem_(np.dot(A.T, -A), A.T, A, 0.001, 20)
    assert_false((Ap < 0).any())

def test_nls_close():
    """
    Test that the nls results should be close
    """
    A = np.atleast_2d(range(1,5))
    Ap, _, _ = nmf._nls_subproblem_(np.dot(A.T, A), A.T, np.zeros_like(A), 
                                    0.001, 20)
    assert_true((np.abs(Ap - A) < 0.01).all())

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
