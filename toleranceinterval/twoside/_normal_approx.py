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

r"""
Algorithms for computing approximate two-sided statistical tolerance interval
factors under the assumption of a normal distribution.

"""

import numpy as np
from scipy.stats import norm, chi2


def tolerance_factor_howe(n, p, g, m=None, nu=None):
    r"""
    Compute two-side central tolerance interval factor using Howe's method.

    Computes the two-sided tolerance interval (TI) factor under a normal
    distribution assumption using Howe's method. This follows the derivation
    in [1]. This is an approximation, and does not represent the exact TI.

    Parameters
    ----------
    n : scalar
        Sample size.
    p : float
        Percentile for central TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.
    m : scalar
        Number of independent random samples (of size n). If None,
        default value is m = 1.
    nu : scalar
        Degrees of freedom for distribution of the (pooled) sample
        variance. If None, default value is nu = m*(n-1).

    Returns
    -------
    float
        The calculated tolerance factor for the tolerance interval.

    References
    ----------
    [1] Howe, W. G. (1969). "Two-sided Tolerance Limits for Normal
        Populations - Some Improvements", Journal of the American Statistical
        Association, 64 , pages 610-620.

    """
    # Handle defaults for keyword inputs.
    if m is None:
        m = 1
    if nu is None:
        nu = m * (n - 1)

    alpha = 1.0 - g
    zp = norm.ppf((1.0 + p) / 2.0)
    u = zp * np.sqrt(1.0 + (1.0 / n))
    chi2_nu = chi2.ppf(alpha, df=nu)
    v = np.sqrt(nu / chi2_nu)
    k = u * v
    return k


def tolerance_factor_guenther(n, p, g, m=None, nu=None):
    r"""
    Compute two-side central tolerance interval factor using Guenther's method.

    Computes the two-sided tolerance interval (TI) factor under a normal
    distribution assumption using Guenthers's method. This follows the
    derivation in [1]. This is an approximation, and does not represent the
    exact TI.

    Parameters
    ----------
    n : scalar
        Sample size.
    p : float
        Percentile for central TI to estimate.
    g : float
        Confidence level where g > 0. and g < 1.
    m : scalar
        Number of independent random samples (of size n). If None,
        default value is m = 1.
    nu : scalar
        Degrees of freedom for distribution of the (pooled) sample
        variance. If None, default value is nu = m*(n-1).

    Returns
    -------
    float
        The calculated tolerance factor for the tolerance interval.

    References
    ----------
    [1] Guenther, W. C. (1977). "Sampling Inspection in Statistical Quality
        Control", Griffin's Statistical Monographs, Number 37, London.

    """
    # Handle defaults for keyword inputs.
    if m is None:
        m = 1
    if nu is None:
        nu = m * (n - 1)

    k = tolerance_factor_howe(n, p, g, m, nu)
    alpha = 1.0 - g
    chi2_nu = chi2.ppf(alpha, df=nu)
    w = np.sqrt(1.0 + ((n - 3.0 - chi2_nu) / (2.0 * (n + 1.0) ** 2)))
    k *= w
    return k
