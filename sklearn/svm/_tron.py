"""
Wrapper for the trust-region optimizer
"""
import numpy as np
from scipy import optimize

from _tron_fast import _fmin_tron

def fmin_tron(func, grad_hess, x0, args=(), max_iter=500, tol=1e-6,
              gtol=1e-3):
    assert callable(func), ("'func' must be callable, %s was given"
                            % func)
    assert callable(grad_hess), (
                    "'grad_hess' must be callable, %s was given"
                    % grad_hess)
    grad, hess = grad_hess(x0)
    assert callable(hess), ("The second argument of 'grad_hess' "
                            "must be callable, %s is output" % grad_hess)
    f0 = func(x0)
    assert np.isscalar(f0), ("'func' must be a real-valued function,"
                             "it returned %s" % f0)
    grad_err = optimize.check_grad(func, lambda x: grad_hess(x)[0],
                                   x0, *args)
    assert grad_err < 1e-4 * f0, "The error on the gradient is too large"

    return _fmin_tron(func, grad_hess, x0, args=args, max_iter=max_iter,
                      tol=tol, gtol=gtol)



