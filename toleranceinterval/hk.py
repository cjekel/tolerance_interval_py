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

from sympy import Symbol, Integral, factorial
from sympy import gamma, hyper, exp_polar, I, pi, log
from scipy.special import betainc, betaincinv
import numpy as np
from warnings import warn


class HansonKoopmans(object):

    def __init__(self, p, g, n, j, method='secant', max_iter=200,
                 tol=1e-5, step_size=1e-4):
        r"""
        An object to solve for the Hanson-Koopmans bound.

        Solve the Hanson-Koopmans [1] bound for any percentile, confidence
        level, and number of samples.  This assumes the lowest value is the
        first order statistic, but you can specify the index of the second
        order statistic as j.

        Parameters
        ----------
        p : float
            Percentile where p < 0.5 and p > 0.
        g : float
            Confidence level where g > 0. and g < 1.
        n : int
            Number of samples.
        j : int
            Index of the second value to use for the second order statistic.
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

        Attributes
        ----------
        b : float_like
            Hanson-Koopmans bound.
        un_conv : bool
            Unconvergence status. If un_conv, then the method did not converge.
        count : int
            Number of iterations used in the root finding method.
        fall_back : bool
            Whether to fall back to the traditional non-parametric method.

        Raises
        ------
        ValueError
            Incorrect input, or unable to comptue the Hanson-Koopmans bound.

        References
        ----------
        [1] Hanson, D. L., & Koopmans, L. H. (1964). Tolerance Limits for
            the Class of Distributions with Increasing Hazard Rates. Ann. Math.
            Statist., 35(4), 1561–1570. https://doi.org/10.1214/aoms/1177700380

        [2] Vangel, M. G. (1994). One-sided nonparametric tolerance limits.
            Communications in Statistics - Simulation and Computation, 23(4),
            1137–1154. https://doi.org/10.1080/03610919408813222

        """
        self.max_iter = max_iter
        self.tol = tol
        self.step_size = step_size
        # create a dummy variable v
        self.v = Symbol('v', nonzero=True, rational=True, positive=True)
        # check that p, g, n, j are valid
        if not(p < 0.5 and p > 0.):
            self.invalid_value(p, 'p')
        else:
            self.p = p
        if not(g > 0. and g < 1.):
            self.invalid_value(g, 'g')
        else:
            self.g = g
        self.n = int(n)
        if not(j < n and j > -1):
            self.invalid_value(j, 'j')
        else:
            self.j = int(j)
        # compute the stuff that doesn't depend on b
        self.constant_vales()
        # compute b = 1
        pi_B_1 = self.piB(0)  # remember that b - 1 = B; b = B + 1
        if pi_B_1 >= self.g:
            self.fall_back = True
            # raise ValueError('b = 1, defer to traditional methods...')
            # raise RunTimeWarning?
        else:
            self.fall_back = False
            b_guess = self.vangel_approx(p=float(self.p))
            # print(float(b_guess))
            if np.isnan(b_guess):
                raise RuntimeError('Bad Vangel Approximation is np.nan')
            elif b_guess <= 0:
                b_guess = 1e-2
                # print(b_guess)
            self.b_guess = b_guess
            if method == 'secant':
                B, status, count = self.secant_solver(b_guess - 1.)
            elif method == 'newton-raphson':
                B, status, count = self.nr_solver(b_guess - 1.)
            elif method == 'halley':
                B, status, count = self.halley_solver(b_guess - 1.)
            else:
                raise ValueError(str(method) + ' is not a valid method!')

            self.b = B + 1.
            self.un_conv = status
            self.count = count
            if self.un_conv:
                war = 'HansonKoopmans root finding method failed to converge!'
                warn(war, RuntimeWarning)
            # This should raise RuntimeError if not converged!

    def invalid_value(self, value, variable):
        err = str(value) + ' was not a valid value for ' + variable
        raise ValueError(err)

    def constant_vales(self):
        self.nj = self.n-self.j-1
        self.A = factorial(self.n) / (factorial(self.nj) *
                                      factorial(self.j-1))
        # compute the left integral
        int_left = (self.p*self.p**self.j*gamma(self.j + 1) *
                    hyper((-self.nj, self.j + 1),
                          (self.j + 2,),
                          self.p*exp_polar(2*I*pi)) /
                    (self.j*gamma(self.j + 2)))
        self.int_left = int_left.evalf()  # evaluates to double precision

    def piB(self, B):
        int_right_exp = (self.v**self.j*(1 - self.v)**self.nj/self.j
                         - (1 - self.v)**self.nj*(-self.p**(1/(B + 1)) *
                         self.v**(B/(B + 1)) + self.v)**self.j/self.j)
        int_right = Integral(int_right_exp, (self.v, self.p, 1)).evalf()
        return (self.int_left + int_right)*self.A

    def dpiB(self, B):
        d_int_right_exp_B = (self.v**self.j*(1 - self.v)**self.nj/self.j -
                             (1 - self.v)**self.nj*(-self.p**(1/(B + 1)) *
                             self.v**(B/(B + 1))*(-B/(B + 1)**2 + 1/(B + 1)) *
                             log(self.v) + self.p**(1/(B + 1))*self.v **
                             (B / (B + 1))*log(self.p)/(B + 1)**2 + self.v) **
                             self.j / self.j)
        d_int_right = Integral(d_int_right_exp_B, (self.v, self.p, 1)).evalf()
        return d_int_right*self.A

    def d2piB2(self, B):
        d2_int_r_B = (-(1 - self.v)**self.nj*(-self.p**(1/(B + 1))*self.v **
                      (B/(B + 1))*(-B/(B + 1)**2 + 1/(B + 1))*log(self.v) +
                      self.p**(1/(B + 1))*self.v**(B/(B + 1))*log(self.p) /
                      (B + 1)**2 + self.v)**self.j*(-self.p**(1/(B + 1)) *
                      self.v**(B/(B + 1))*(2*B/(B + 1)**3 - 2/(B + 1)**2) *
                      log(self.v) - self.p**(1/(B + 1))*self.v**(B/(B + 1)) *
                      (-B/(B + 1)**2 + 1/(B + 1))**2*log(self.v)**2 + 2 *
                      self.p**(1/(B + 1))*self.v**(B/(B + 1)) *
                      (-B/(B + 1)**2 + 1/(B + 1))
                      * log(self.p)*log(self.v)/(B + 1)**2 - 2 *
                      self.p**(1/(B + 1))*self.v**(B/(B + 1))*log(self.p) /
                      (B + 1)**3 - self.p**(1/(B + 1))*self.v**(B/(B + 1)) *
                      log(self.p)**2/(B + 1)**4)/(-self.p**(1/(B + 1)) *
                      self.v**(B/(B + 1))*(-B/(B + 1)**2 + 1/(B + 1)) *
                      log(self.v) + self.p**(1/(B + 1))*self.v **
                      (B/(B + 1)) *
                      log(self.p)/(B + 1)**2 + self.v))
        d2_int_right = Integral(d2_int_r_B, (self.v, self.p, 1)).evalf()
        return d2_int_right*self.A

    def vangel_approx(self, n=None, i=None, j=None, p=None, g=None):
        if n is None:
            n = self.n
        if i is None:
            i = 1
        if j is None:
            j = self.j+1
        if p is None:
            p = self.p
        if g is None:
            g = self.g
        betatmp = betainc(j, n-j+1, p)
        a = g - betatmp
        b = 1.0 - betatmp
        q = betaincinv(i, j-i, a/b)
        return np.log(((p)*(n+1))/j) / np.log(q)

    def secant_solver(self, B_guess, max_iter=None, tol=None, step_size=None):
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        if step_size is None:
            step_size = self.step_size
        count = 0
        f = self.piB(B_guess) - self.g
        f1 = self.piB(B_guess + step_size) - self.g
        dfdx = (f1 - f) / step_size
        B_next = B_guess - (f/dfdx)
        un_conv = np.abs(B_next - B_guess) > tol
        while un_conv and count < max_iter:
            B_guess = B_next
            f = self.piB(B_guess) - self.g
            f1 = self.piB(B_guess + step_size) - self.g
            dfdx = (f1 - f) / step_size
            B_next = B_guess - (f/dfdx)
            un_conv = np.abs(B_next - B_guess) > tol
            count += 1
        return B_next, un_conv, count

    def nr_solver(self, B_guess, max_iter=None, tol=None):
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        count = 0
        f = self.piB(B_guess) - self.g
        dfdx = self.dpiB(B_guess)
        B_next = B_guess - (f/dfdx)
        un_conv = np.abs(B_next - B_guess) > tol
        while un_conv and count < max_iter:
            B_guess = B_next
            f = self.piB(B_guess) - self.g
            dfdx = self.dpiB(B_guess)
            B_next = B_guess - (f/dfdx)
            un_conv = np.abs(B_next - B_guess) > tol
            count += 1
        return B_next, un_conv, count

    def halley_solver(self, B_guess, max_iter=None, tol=None):
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        count = 0
        f = self.piB(B_guess) - self.g
        dfdx = self.dpiB(B_guess)
        d2fdx2 = self.d2piB2(B_guess)
        B_next = B_guess - ((2*f*dfdx) / (2*(dfdx**2) - (f*d2fdx2)))
        un_conv = np.abs(B_next - B_guess) > tol
        while un_conv and count < max_iter:
            B_guess = B_next
            f = self.piB(B_guess) - self.g
            dfdx = self.dpiB(B_guess)
            d2fdx2 = self.d2piB2(B_guess)
            B_next = B_guess - ((2*f*dfdx) / (2*(dfdx**2) - (f*d2fdx2)))
            un_conv = np.abs(B_next - B_guess) > tol
            count += 1
        return B_next, un_conv, count
