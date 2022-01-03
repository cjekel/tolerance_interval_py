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
from ..checks import numpy_array, assert_2d_sort
from . import _normal_exact as exact
from . import _normal_approx as approx


def normal_factor(n, p, g, method=None, m=None, nu=None, d2=None,
                  simultaneous=False, tailprob=False):
    r"""
    Compute two-sided central tolerance factor using the normal distribution.

    Computes the tolerance factor k for the two-sided central tolerance
    interval (TI) to cover a proportion p of the population with confidence g:

        TI = [Xmean - k * S, Xmean + k * S]

    where Xmean = mean(X), S = std(X), X = [X_1,...,X_n] is a random sample
    of size n from the distribution N(mu,sig2) with unknown mean mu and
    variance sig2.

    The tolerance factor k is determined such that the tolerance intervals
    with confidence g cover at least the coverage fraction
    of the distribution N(mu,sigma^2), i.e.
    Prob[ Prob( Xmean - k * S < X < Xmean + k * S ) >= p ] = g,
    for X ~ N(mu,sig2) which is independent of Xmean and S.

    By default, this function uses an 'exact' method for computing the factor
    by Gauss-Kronod quadrature as described in the references [1,2,4]. There
    are also two approximate methods implemented: the 'howe' method as
    described in [5], and the 'guenther' method as described in [6]. A brief
    overview of both approximate methods can be found at NIST's website:
    https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm

    Additional optional parameters are available to consider pooled variance
    studies when m random samples of size n are available. Furthermore,
    for the 'exact' method, optional parameters are available to
    consider simultaneous tolerance intervals as described in [7,8].
    If S is a pooled estimator of sig, based on m random samples of size n,
    normal_factor computes the tolerance factor k for the two-sided p-content
    and g-confidence tolerance intervals

        TI = [Xmean_i - k * S, Xmean_i + k * S], for i = 1,...,m

    where Xmean_i = mean(X_i), X_i = [X_i1,...,X_in] is a random sample of
    size n from the distribution N(mu_i,sig2) with unknown mean mu_i and
    variance sig2, and S = sqrt(S2), S2 is the pooled estimator of sig2,

        S2 = (1/nu) * sum_i=1:m ( sum_j=1:n (X_ij - Xmean_i)^2 )

    with nu degrees of freedom, nu = m * (n-1). For the 'exact' method, both
    the simultaneous and non-simultaneous cases can be considered.

    Parameters
    ----------
    n : scalar
        Sample size
    p : scalar in the interval [0.0, 1.0]
        Coverage (or content) probability,
        Prob( Xmean - k * S < X < Xmean + k * S ) >= p
    g : scalar in the interval [0.0, 1.0]
        Confidence probability,
        Prob[ Prob( Xmean-k*S < X < Xmean+k*S ) >= p ] = g.
    method : str
        Method to use for computing the factor. Available methods are 'exact',
        'howe', and 'guenther'. If None, the default method is 'exact'.
    m : scalar
        Number of independent random samples (of size n). If None,
        default value is m = 1.
    nu : scalar
        Degrees of freedom for distribution of the (pooled) sample
        variance S2. If None, default value is nu = m*(n-1).
    d2 : scalar
        Normalizing constant. For computing the factors of the
        non-simultaneous tolerance limits (xx'*betaHat +/- k * S)
        for the linear regression y = XX*beta +epsilon, set d2 =
        xx'*inv(XX'*XX)*xx.
        Typically, in simple linear regression the estimator S2 has
        nu = n-2 degrees of freedom. If None, default value is d2 = 1/n.
    simultaneous : boolean
        Logical flag for calculating the factor for
        simultaneous tolerance intervals. If False, normal_factor will
        calculate the factor for the non-simultaneous tolerance interval.
        Default value is False.
    tailprob : boolean
        Logical flag for representing the input probabilities
        'p' and 'g'. If True, the input parameters are
        represented as the tail coverage (i.e. 1 - p) and tail confidence
        (i.e. 1 - g). This option is useful if the interest is to
        calculate the tolerance factor for extremely large values
        of coverage and/or confidence, close to 1, as
        e.g. coverage = 1 - 1e-18. Default value is False.

    Returns
    -------
    float
        The calculated tolerance factor for the tolerance interval.

    References
    ----------
    [1] Krishnamoorthy K, Mathew T. (2009). Statistical Tolerance Regions:
        Theory, Applications, and Computation. John Wiley & Sons, Inc.,
        Hoboken, New Jersey. ISBN: 978-0-470-38026-0, 512 pages.

    [2] Witkovsky V. On the exact two-sided tolerance intervals for
        univariate normal distribution and linear regression. Austrian
        Journal of Statistics 43(4), 2014, 279-92.
        http://ajs.data-analysis.at/index.php/ajs/article/viewFile/vol43-4-6/35

    [3] ISO 16269-6:2014: Statistical interpretation of data - Part 6:
        Determination of statistical tolerance intervals.

    [4] Janiga I., Garaj I.: Two-sided tolerance limits of normal
        distributions with unknown means and unknown common variability.
        MEASUREMENT SCIENCE REVIEW, Volume 3, Section 1, 2003, 75-78.

    [5] Howe, W. G. “Two-Sided Tolerance Limits for Normal Populations,
        Some Improvements.” Journal of the American Statistical Association,
        vol. 64, no. 326, [American Statistical Association, Taylor & Francis,
        Ltd.], 1969, pp. 610–20, https://doi.org/10.2307/2283644.

    [6] Guenther, W. C. (1977). "Sampling Inspection in Statistical Quality
        Control",, Griffin's Statistical Monographs, Number 37, London.

    [7] Robert W. Mee (1990) Simultaneous Tolerance Intervals for Normal
        Populations With Common Variance, Technometrics, 32:1, 83-92,
        DOI: 10.1080/00401706.1990.10484595

    [8] K. Krishnamoorthy & Saptarshi Chakraberty (2022). Construction of
        simultaneous tolerance intervals for several normal distributions,
        Journal of Statistical Computation and Simulation, 92:1, 101-114,
        DOI: 10.1080/00949655.2021.1932885

    """
    # Handle default method:
    if method is None:
        method = 'exact'

    if method == 'exact':
        k = exact.tolerance_factor(n, p, g, m, nu, d2, simultaneous, tailprob)
    elif method == 'howe':
        k = approx.tolerance_factor_howe(n, p, g, m, nu)
    elif method == 'guenther':
        k = approx.tolerance_factor_guenther(n, p, g, m, nu)
    else:
        raise ValueError(
            "Invalid method requested. Valid methods are 'exact', 'howe', or "
            "'guenther'."
        )
    return k


def normal(x, p, g, method=None, pool_variance=False):
    r"""
    Compute two-sided central tolerance interval using the normal distribution.

    Computes the two-sided tolerance interval (TI) to cover a proportion p of
    the population with confidence g using the normal distribution. This
    follows the standard approach to calculate the interval as a factor of
    sample standard deviations away from the sample mean.

        TI = [Xmean - k * S, Xmean + k * S]

    where Xmean = mean(X), S = std(X), X = [X_1,...,X_n] is a random sample
    of size n from the distribution N(mu,sig2) with unknown mean mu and
    variance sig2.

    By default, this function uses an 'exact' method for computing the TI
    by Gauss-Kronod quadrature. There are also two approximate methods
    implemented: the 'howe' method, and the 'guenther' method. See the
    documentation for normal_factor for more details on the methods.

    Parameters
    ----------
    x : ndarray (1-D, or 2-D)
        Numpy array of samples to compute the tolerance interval. Assumed data
        type is np.float. Shape of (m, n) is assumed for 2-D arrays with m
        number of sets of sample size n.
    p : float
        Percentile for central TI to cover.
    g : float
        Confidence level where g > 0. and g < 1.
    method : str
        Method to use for computing the TI. Available methods are 'exact',
        'howe', and 'guenther'. If None, the default method is 'exact'.
    pool_variance : boolean
        Consider the m random samples to share the same variance such that
        the degrees of freedom are nu = m*(n-1). Default is False.

    Returns
    -------
    ndarray (2-D)
        The normal distribution toleranace interval bound. Shape (m, 2) from m
        sets of samples, where [:, 0] is the lower bound and [:, 1] is the
        upper bound.

    References
    ----------
    See the documentation for normal_factor for a complete list of references.

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

    # Handle pooled variance case
    if pool_variance:
        _m = m
    else:
        _m = 1

    k = normal_factor(n, p, g, method, _m)
    bound = np.zeros((m, 2))
    xmu = x.mean(axis=1)
    kstd = k * x.std(axis=1, ddof=1)
    bound[:, 0] = xmu - kstd
    bound[:, 1] = xmu + kstd
    return bound


def lognormal(x, p, g, method=None, pool_variance=False):
    r"""
    Two-sided central tolerance interval using the lognormal distribution.

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
    method : str
        Method to use for computing the TI. Available methods are 'exact',
        'howe', and 'guenther'. If None, the default method is 'exact'.
    pool_variance : boolean
        Consider the m random samples to share the same variance such that
        the degrees of freedom are nu = m*(n-1). Default is False.

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
    return np.exp(normal(np.log(x), p, g, method, pool_variance))
