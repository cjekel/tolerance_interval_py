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
from scipy.stats import binom, norm, nct
from .hk import HansonKoopmans
from .checks import numpy_array_sort


def normal(x, p, g):
    r"""
    Compute tolerance interval using the normal distribution.

    Computes the one-sided tolerance interval using the normal distribution.
    This follows the derivation in [1] to calculate the interval as a factor
    of sample standard deviations away from the sample mean.

    Parameters
    ----------
    x : ndarray (1-D)
        Numpy array of samples to compute the tolerance interval. Assumed data
        type is np.float.
    p : float
        Percentile for the TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.

    Returns
    -------
    float
        The normal distribution toleranace interval bound.

    References
    ----------
    .. [1] Young, D. S. (2010). tolerance: An R Package for Estimating
        Tolerance Intervals. Journal of Statistical Software; Vol 1, Issue 5
        (2010). Retrieved from http://dx.doi.org/10.18637/jss.v036.i05

    Examples
    --------
    Estimate the 10th percentile with 95% confidence of the following 100
    random samples from a normal distribution.

    >>> import numpy as np
    >>> import toleranceinterval
    >>> x = np.random.nomral(100)
    >>> bound = toleranceinterval.normal(x, 0.1, 0.95)

    """
    x, n = numpy_array_sort(x)
    if p < 0.5:
        p = 1.0 - p
        minus = True
    else:
        minus = False
    zp = norm.ppf(p)
    t = nct.ppf(1.0-g, df=n-1., nc=np.sqrt(n)*zp)
    k = t / np.sqrt(n)
    if minus:
        return x.mean() - (k*x.std(ddof=1))
    else:
        return x.mean() + (k*x.std(ddof=1))


def non_parametric(x, p, g):
    r"""
    Compute tolerance interval using the traditional non-parametric method.

    Computes a tolerance interval for any percentile, confidence level, and
    number of samples using the traditional non-parametric method [1] [2].
    This assumes that the true distribution is continuous.

    Parameters
    ----------
    x : ndarray (1-D)
        Numpy array of samples to compute the non-parametric tolerance
        interval. Assumed data type is np.float.
    p : float
        Percentile for the TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.

    Returns
    -------
    float
        The non-parametric toleranace interval bound. Returns np.nan if a
        non-parametric tolerance interval does not exist for the combination
        of percentile, confidence level, and number of samples.

    Notes
    -----
    The non-parametric tolerance inteval only exists for certain combinations
    of percentile, confidence level, and number of samples.

    References
    ----------
    .. [1] Hong, L. J., Huang, Z., & Lam, H. (2017). Learning-based robust
        optimization: Procedures and statistical guarantees. ArXiv Preprint
        ArXiv:1704.04342.
    .. [2] 9.5.5.3 Nonparametric Procedure. (2017). In MMPDS-12 : Metallic
        materials properties development and standardization. Battelle
        Memorial Institute.

    Examples
    --------
    Estimate the 10th percentile with 95% confidence of the following 300
    random samples from a normal distribution.

    >>> import numpy as np
    >>> import toleranceinterval
    >>> x = np.random.random(300)
    >>> bound = toleranceinterval.normal(x, 0.1, 0.95)

    """
    x, n = numpy_array_sort(x)
    r = np.arange(0, n)
    confidence_index = binom.sf(r, n, p)
    boolean_index = confidence_index >= g
    if boolean_index.sum() > 0:
        return x[np.max(np.where(boolean_index))]
    else:
        return np.nan


def hanson_koopmans(x, p, g, j=-1, method='secant', max_iter=200, tol=1e-5,
                    step_size=1e-4):
    r"""
    Compute left tail probabilities using the HansonKoopmans method [1].

    Runs the HansonKoopmans solver object to find the left tail bound for any
    percentile, confidence level, and number of samples. This assumes the
    lowest value is the first order statistic, but you can specify the index
    of the second order statistic as j.

    Parameters
    ----------
    x : ndarray (1-D)
        Numpy array of samples to compute the HansonKoopmans tolerance
        interval. Assumed data type is np.float.
    p : float
        Percentile where p < 0.5 and p > 0.
    g : float
        Confidence level where g > 0. and g < 1.
    n : int
        Number of samples.
    j : int, optional
        Index of the second value to use for the second order statistic.
        Default is the last value j = -1 = n-1.
    method : string, optional
        Which rootfinding method to use to solve for the Hanson-Koopmans
        bound. Default is method='secant' which appears to converge
        quickly. Other choices include 'newton-raphson' and 'halley'.
    max_iter : int, optional
        Maximum number of iterations for the root finding method.
    tol : float, optional
        Tolerance for the root finding method to converge.
    step_size : float, optional
        Step size for the secant solver. Default step_size = 1e-4.

    Returns
    -------
    float
        The Hanson-Koopmans toleranace interval bound. Returns np.nan if the
        rootfinding method did not converge.

    Notes
    -----
    The Hanson-Koopmans bound assumes the true distribution belongs to the
    log-concave CDF class of distributions [1].

    This implemnation will always extrapolate beyond the lowest sample. If
    interpolation is needed within the sample set, this method falls back to
    the traditional non-parametric method using non_parametric(x, p, g).

    j uses Python style index notation.


    References
    ----------
    .. [1] Hanson, D. L., & Koopmans, L. H. (1964). Tolerance Limits for
        the Class of Distributions with Increasing Hazard Rates. Ann. Math.
        Statist., 35(4), 1561–1570. https://doi.org/10.1214/aoms/1177700380

    Examples
    --------
    Estimate the 10th percentile with 95% confidence of the following 10
    random samples.

    >>> import numpy as np
    >>> import toleranceinterval
    >>> x = np.random.random(10)
    >>> bound = toleranceinterval.hanson_koopmans(x, 0.1, 0.95)

    """
    x, n = numpy_array_sort(x)
    if j == -1:
        # Need to use n for the HansonKoopmans solver
        j = n - 1
    assert j < n
    myhk = HansonKoopmans(p, g, n, j)
    if myhk.fall_back:
        return non_parametric(x, p, g)
    if myhk.un_conv:
        return np.nan
    else:
        b = float(myhk.b)
        bound = x[j] * (x[0]/x[j])**b
        return bound
