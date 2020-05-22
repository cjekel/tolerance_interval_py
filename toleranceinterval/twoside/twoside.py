# -- coding: utf-8 --
# MIT License
#
# Copyright (c) 2019 Charles Jekel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.stats import norm, chi2
from ..checks import numpy_array, assert_2d_sort


def normal(x, p, g):
    r"""
    Compute two-side central tolerance interval using the normal distribution.

    Computes the two-sided tolerance interval (TI) using the normal
    distribution. This follows the derivation in [1] to calculate the interval
    as a factor of sample standard deviations away from the sample mean. This
    is an approximation, and does not represent the exact TI. For exact TI see
    the eqution 2.3.4 from [2].

    Parameters
    ----------
    x : ndarray (1-D, or 2-D)
        Numpy array of samples to compute the tolerance interval. Assumed data
        type is np.float. Shape of (m, n) is assumed for 2-D arrays with m
        number of sets of sample size n.
    p : float
        Percentile for central TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.

    Returns
    -------
    ndarray (2-D)
        The normal distribution toleranace interval bound. Shape (m, 2) from m
        sets of samples, where [:, 0] is the lower bound and [:, 1] is the
        upper bound.

    References
    ----------
    .. [1] Young, D. S. (2010). tolerance: An R Package for Estimating
        Tolerance Intervals. Journal of Statistical Software; Vol 1, Issue 5
        (2010). Retrieved from http://dx.doi.org/10.18637/jss.v036.i05
    .. [2] Krishnamoorthy, K. and Mathew, T. (2009). Statistical Tolerance
        Regions: Theory, Applications, and Computation. Wiley, Hoboken, NJ.

    Examples
    --------
    Estimate the 90th percentile central TI with 95% confidence of the
    following 100 random samples from a normal distribution.

    >>> import numpy as np
    >>> import toleranceinterval as ti
    >>> x = np.random.nomral(100)
    >>> bound = ti.twoside.normal(x, 0.9, 0.95)
    >>> print('Lower bound:', bound[:, 0])
    >>> print('Upper bound:', bound[:, 1])

    Estimate the 95th percentile central TI with 95% confidence of the
    following 100 random samples from a normal distribution.

    >>> bound = ti.twoside.normal(x, 0.95, 0.95)

    """
    x = numpy_array(x)  # check if numpy array, if not make numpy array
    x = assert_2d_sort(x)
    m, n = x.shape
    alpha = 1.0 - g
    zp = norm.ppf((1.0+p)/2.)
    u = zp * np.sqrt(1 + (1.0/n))
    chi2v = chi2.ppf(alpha, df=n-1.0)
    v = np.sqrt((n-1.0)/chi2v)
    w = np.sqrt(1.0 + ((n-3.0-chi2v)/(2.0*(n+1.0)**2)))
    k = u * v * w
    bound = np.zeros((m, 2))
    xmu = x.mean(axis=1)
    kstd = k*x.std(axis=1, ddof=1)
    bound[:, 0] = xmu - kstd
    bound[:, 1] = xmu + kstd
    return bound


def lognormal(x, p, g):
    r"""
    Two-side central tolerance interval using the lognormal distribution.

    Computes the two-sided tolerance interval using the lognormal distribution.
    This just performs a ln and exp transformations of the normal distribution.

    Parameters
    ----------
    x : ndarray (1-D, or 2-D)
        Numpy array of samples to compute the tolerance interval. Assumed data
        type is np.float. Shape of (m, n) is assumed for 2-D arrays with m
        number of sets of sample size n.
    p : float
        Percentile for central TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.

    Returns
    -------
    ndarray (2-D)
        The lognormal distribution toleranace interval bound. Shape (m, 2)
        from m sets of samples, where [:, 0] is the lower bound and [:, 1] is
        the upper bound.

    Examples
    --------
    Estimate the 90th percentile central TI with 95% confidence of the
    following 100 random samples from a lognormal distribution.

    >>> import numpy as np
    >>> import toleranceinterval as ti
    >>> x = np.random.random(100)
    >>> bound = ti.twoside.lognormal(x, 0.9, 0.95)
    >>> print('Lower bound:', bound[:, 0])
    >>> print('Upper bound:', bound[:, 1])

    Estimate the 95th percentile central TI with 95% confidence of the
    following 100 random samples from a normal distribution.

    >>> bound = ti.twoside.lognormal(x, 0.95, 0.95)

    """
    x = numpy_array(x)  # check if numpy array, if not make numpy array
    x = assert_2d_sort(x)
    return np.exp(normal(np.log(x), p, g))
