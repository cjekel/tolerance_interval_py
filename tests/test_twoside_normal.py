# -- coding: utf-8 --
import numpy as np
from toleranceinterval.twoside import normal
# from scipy.stats import chi2
import unittest


class TestEverything(unittest.TestCase):

    def _run_test_cases(self, N, P, G, K, method):
        for i, k in enumerate(K):
            n = N[i]
            x = np.random.random(n) * 10
            xmu = x.mean()
            xstd = x.std(ddof=1)
            p = P[i]
            g = G[i]
            bound = normal(x, p, g, method=method)
            k_hat_l = (xmu - bound[0, 0]) / xstd
            k_hat_u = (bound[0, 1] - xmu) / xstd
            self.assertTrue(np.isclose(k, k_hat_l, rtol=1e-4, atol=1e-5))
            self.assertTrue(np.isclose(k, k_hat_u, rtol=1e-4, atol=1e-5))

    def test_exact(self):
        # We test only one case here mostly as a sanity check to make sure
        # the functions are hooked up correctly. See two-sided tolerance
        # factor unit tests for exhaustive checking of exact tolerance factors.
        # Test case is drawn from ISO Table F1.
        G = [0.90]
        N = [35]
        P = [0.90]
        K = [1.9906]
        self._run_test_cases(N, P, G, K, 'exact')

    def test_guenther_approx(self):
        # values from:
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tolerance_Intervals_for_Normal_Data.pdf
        G = [0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
             0.95]
        N = [26, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 5910, 866, 179]
        P = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        K = [1.6124, 1.7984, 1.7493, 1.7287, 1.7168, 1.7088, 1.7029, 1.6984,
             1.6948, 1.6703, 1.7138, 1.8084]
        self._run_test_cases(N, P, G, K, 'guenther')

    def test_howe_approx(self):
        # values from:
        # https://www.itl.nist.gov/div898/handbook/prc/section2/prc263.htm
        # Howe's method is implicitly tested to some extent by Guenther's
        # method.
        G = [0.99]
        N = [43]
        P = [0.9]
        K = [2.2173]
        self._run_test_cases(N, P, G, K, 'howe')

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        for m in M:
            for n in N:
                x = np.random.random((m, n))
                bounds = normal(x, 0.95, 0.95)
                _m, _ = bounds.shape
                self.assertTrue(_m == m)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            normal(x, 0.9, 0.9)


if __name__ == '__main__':
    np.random.seed(121)
    unittest.main()
