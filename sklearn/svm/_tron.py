"""
Wrapper for the trust-region optimizer
"""
import numpy as np
from scipy import optimize

from _tron_fast import _fmin_tron


def fmin_tron(func, grad_hess, x0, args=(), max_iter=500, tol=1e-6,
              gtol=1e-3):
    """minimize func using Trust Region Newton algorithm

    Parameters
    ----------
    func : callable
        func(w, *args) is the evaluation of the function at w, It
        should return a float.
    grad_hess: callable
        TODO
    x0 : array
        starting point for iteration.
    gtol: float
        stopping criterion. Gradient norm must be less than gtol
        before succesful termination.
    check_inputs: boolean, optional
        If check_inputs is True, basic checks are performed on inputs.
        Set check_inputs to false only if you know what you are doing

    Returns
    -------
    x0 : array
        The minimizer
    res : dict
        A dictionary giving more details on the optimization
    """
    assert callable(func), ("'func' must be callable, %s was given"
                            % func)
    assert callable(grad_hess), (
                    "'grad_hess' must be callable, %s was given"
                    % grad_hess)
    grad, hess = grad_hess(x0, *args)
    assert callable(hess), ("The second argument of 'grad_hess' "
                            "must be callable, %s is output" % grad_hess)
    f0 = func(x0, *args)
    assert np.isscalar(f0), ("'func' must be a real-valued function,"
                             "it returned %s" % f0)
    grad_err = optimize.check_grad(lambda x: func(x, *args),
                                   lambda x: grad_hess(x, *args)[0],
                                   x0)
    assert grad_err < 1e-4 * f0, "The error on the gradient is too large"

    return _fmin_tron(func, grad_hess, x0, args=args, max_iter=max_iter,
                      tol=tol, gtol=gtol)



