# Author: Copyright (c) 2021 Jed Ludlow
# License: MIT License

r"""
Algorithm for computing the exact two-sided statistical tolerance interval
factor under the assumption of a normal distribution.

This module is a Python port of an algorithm written in MATLAB by
Viktor Witkovsky and posted to the MATLAB Central File Exchange, retrieved
2021-03-12 at this URL:

https://www.mathworks.com/matlabcentral/fileexchange/24135-tolerancefactor

Here is Witkovsky's original copyright and disclaimer:

-------------------------------------------------------------------------------
Copyright (c) 2020, Viktor Witkovsky
Copyright (c) 2013, Viktor Witkovsky
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of Institute of Measurement Science, Slovak Academy of
  Sciences, Bratislava, Slovakia nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.stats as stats
import scipy.special as spec
import scipy.integrate as integ
import scipy.optimize as optim

SQRT_2 = 1.4142135623730950488
SQRT_2PI = 2.5066282746310005024


def tolerance_factor(n, coverage, confidence, m=None, nu=None, d2=None,
                     simultaneous=False, tailprob=False):
    r"""
    Compute the exact tolerance factor k for the two-sided statistical
    tolerance interval to contain a proportion of a population with given
    confidence under normal distribution assumptions.

    Computes (by Gauss-Kronod quadrature) the exact tolerance factor k for the
    two-sided coverage-content and (1-alpha)-confidence tolerance interval

        TI = [Xmean - k * S, Xmean + k * S]

    where Xmean = mean(X), S = std(X), X = [X_1,...,X_n] is a random sample
    of size n from the distribution N(mu,sig2) with unknown mean mu and
    variance sig2.

    The tolerance factor k is determined such that the tolerance intervals
    with confidence (1-alpha) cover at least the coverage fraction
    of the distribution N(mu,sigma^2), i.e.
    Prob[ Prob( Xmean - k * S < X < Xmean + k * S ) >= p ]= 1-alpha,
    for X ~ N(mu,sig2) which is independent with Xmean and S. For more
    details see e.g. Krishnamoorthy and Mathew (2009).

    REMARK:
    If S is a pooled estimator of sig, based on m random samples of size n,
    tolerance_factor computes the simultaneous (or non-simultaneous) exact
    tolerance factor k for the two-sided coverage-content and
    (1-alpha)-confidence tolerance intervals

        TI = [Xmean_i - k * S, Xmean_i + k * S], for i = 1,...,m

    where Xmean_i = mean(X_i), X_i = [X_i1,...,X_in] is a random sample of
    size n from the distribution N(mu,sig2) with unknown mean mu and
    variance sig2, and S = sqrt(S2), S2 is the pooled estimator of sig2,

        S2 = (1/nu) * sum_i=1:m ( sum_j=1:n (X_ij - Xmean_i)^2 )

    with nu degrees of freedom, nu = m * (n-1).

    Parameters
    ----------
    n : scalar
        Sample size
    coverage : scalar in the interval [0.0, 1.0]
        Coverage (or content) probability,
        Prob( Xmean - k * S < X < Xmean + k * S ) >= coverage
    confidence : scalar in the interval [0.0, 1.0]
        Confidence probability,
        Prob[ Prob( Xmean-k*S < X < Xmean+k*S ) >= coverage ] = confidence.
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
        simultaneous tolerance intervals. If False, tolerance_factor will
        calculate the factor for the non-simultaneous tolerance interval.
        Default value is False.
    tailprob : boolean
        Logical flag for representing the input probabilities
        'coverage' and 'confidence'. If True, the input parameters are
        represented as the tail coverage (i.e. 1 - coverage) and tail
        confidence (i.e. 1 - confidence). This option is useful if the
        interest is to calculate the tolerance factor for extremely large
        values of coverage and/or confidence, close to 1, as e.g.
        coverage = 1 - 1e-18. Default value is False.

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

    [5] Robert W. Mee (1990) Simultaneous Tolerance Intervals for Normal
        Populations With Common Variance, Technometrics, 32:1, 83-92,
        DOI: 10.1080/00401706.1990.10484595

    """
    # Coverage and confidence must be within [0.0, 1.0].
    if (coverage < 0.0) or (coverage > 1.0):
        raise ValueError('coverage must be within the interval [0.0, 1.0].')
    if (confidence < 0.0) or (confidence > 1.0):
        raise ValueError('confidence must be within the interval [0.0, 1.0].')

    # Handle defaults for keyword inputs.
    if m is None:
        m = 1
    if nu is None:
        nu = m * (n - 1)
    if d2 is None:
        d2 = 1.0 / n

    # Set values for limiting cases of coverage and confidence.
    if confidence == 0:
        k = np.nan
        return k
    if confidence == 1:
        k = np.inf
        return k
    if coverage == 1:
        k = np.inf
        return k
    if coverage == 0:
        k = 0.0
        return k

    # Set the constants.
    if tailprob:
        tailconfidence = confidence
        tailcoverage = coverage
    else:
        tailconfidence = round(1.0e+16 * (1.0 - confidence)) / 1.0e+16
        tailcoverage = round(1.0e+16 * (1.0 - coverage)) / 1.0e+16

    # Return the result for nu = Inf.
    if nu == np.inf:
        k = SQRT_2 * spec.erfcinv(tailcoverage)
        return k

    # Handle the case of nu not being positive.
    if nu <= 0:
        raise(ValueError, 'Degrees of freedom should be positive.')

    # Compute the two-sided tolerance factor
    tol_high_precision = np.spacing(tailconfidence)

    # Integration limits
    A = 0.0
    B = 10.0

    if m > 1:
        k0, _ = _approx_tol_factor_witkovsky(
            tailcoverage, tailconfidence, d2, m, nu, A, B)
    else:
        k0, _ = _approx_tol_factor_wald_wolfowitz(
            tailcoverage, tailconfidence, d2, nu)

    # Compute the tolerance factor.
    sol = optim.newton(lambda k: (_integral_gauss_kronod(
        k, nu, m, d2, tailcoverage, simultaneous, A, B, tol_high_precision)
        - tailconfidence), k0)
    k = sol

    return k


def _integral_gauss_kronod(k, nu, m, c, tailcoverage, simultaneous, A, B, tol):
    r"""
    Evaluates the integral defined by eqs. (1.2.4) and (2.5.8) in
    Krishnamoorthy and Mathew: Statistical Tolerance Regions, Wiley, 2009,
    (pp.7 and 52), by adaptive Gauss-Kronod quadrature.

    (The tol argument is currently ignored but may be used at some point. It
    was used in Witkovsky's integration, but we ignore it here and use
    SciPy default tolerances, which seem to be adequate.)

    """
    val, _ = integ.quad(lambda z: _integrand(
        z, k, nu, m, c, tailcoverage, simultaneous), A, B)
    val *= 2
    return val


def _integrand(z, k, nu, m, c, tailcoverage, simultaneous):
    r"""
    Integrand for Gauss-Kronod quadrature.

    """
    root = _find_root(np.sqrt(c) * z, tailcoverage)
    ncx2pts = nu * root**2
    factor = np.exp(-0.5 * z**2) / SQRT_2PI

    if simultaneous:
        factor = factor * (m * (1 - (spec.erfc(z / SQRT_2))) ** (m - 1))

    x = ncx2pts / k ** 2
    fun = spec.gammainc(nu / 2.0, x / 2.0) * factor
    return fun


def _find_root(x, tailcoverage):
    r"""
    Numerically finds the solution (root), of the equation

    normcdf(x+root) - normcdf(x-root) = coverage = 1 - tailcoverage

    by Halley's method for finding the root of the function

    fun(r) = fun(r|x,tailcoverage)

    based on two derivatives, funD1(r|x,tailcoverage)
    and funD2(r|x,tailcoverage) of the fun(r|x,tailcoverage), (for given x
    and tailcoverage).

    Note that r = sqrt(ncx2inv(1-tailcoverage,1,x^2)), where ncx2inv is the
    inverse of the noncentral chi-square cdf.

    """
    # Set the constants
    max_iterations = 100
    iteration = 0

    # Set the appropriate tolerance
    if np.spacing(tailcoverage) < np.spacing(1):
        tol = min(10 * np.spacing(tailcoverage), np.spacing(1))

    # Set the starting value of the root r: r0 = x + norminv(coverage)
    r = x + SQRT_2 * spec.erfcinv(2 * tailcoverage)

    # Main loop (Halley's method)
    while True:
        iteration += 1
        fun, fun_d1, fun_d2 = _complementary_content(r, x, tailcoverage)
        # Halley's method
        r = r - 2 * fun * fun_d1 / (2 * fun_d1 ** 2 - fun * fun_d2)
        if iteration > max_iterations:
            break
        if np.all(np.abs(fun) < tol):
            break
    return r


def _complementary_content(r, x, tailcoverage):
    r"""
    Complementary content function.

    Calculates difference between the complementary content and given
    the tailcoverage

    fun(r|x,tailcoverage) = 1 - (normcdf(x+r) - normcdf(x-r)) - tailcoverage

    and the first (fun_d1) and the second (fun_d2) derivative of the function

    fun(r|x,tailcoverage)

    """
    fun = 0.5 * (
        spec.erfc((x + r) / SQRT_2)
        + spec.erfc(-(x - r) / SQRT_2)
    ) - tailcoverage
    aux1 = np.exp(-0.5 * (x + r)**2)
    aux2 = np.exp(-0.5 * (x - r)**2)
    fun_d1 = -(aux1 + aux2) / SQRT_2PI
    fun_d2 = -((x - r) * aux2 - (x + r) * aux1) / SQRT_2PI
    return (fun, fun_d1, fun_d2)


def _approx_tol_factor_wald_wolfowitz(tailcoverage, tailconfidence, c, nu):
    r"""
    Compute the approximate tolerance factor (Wald-Wolfowitz).

    """
    r = _find_root(np.sqrt(c), tailcoverage)
    k = r * np.sqrt(nu / stats.chi2.ppf(tailconfidence, nu))
    return (k, r)


def _approx_tol_factor_witkovsky(tailcoverage, tailconfidence, c, m, nu, A, B):
    r"""
    Compute the approximate tolerance factor (Witkovsky).

    """
    val, _ = integ.quad(lambda z: _expect_fun(z, c, tailcoverage, m), A, B)
    r = np.sqrt(2 * val)
    k = r * np.sqrt(nu / stats.chi2.ppf(tailconfidence, nu))
    return (k, r)


def _expect_fun(z, c, tailcoverage, m):
    r"""
    Expectation function.

    """
    r = _find_root(np.sqrt(c) * z, tailcoverage)
    f = r ** 2 * stats.norm.pdf(z)
    if m > 1:
        f = f * (m * (1 - (spec.erfc(z / SQRT_2))) ** (m - 1))
    return f
