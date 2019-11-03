# -- coding: utf-8 --
import numpy as np
from toleranceinterval.twoside import lognormal
# from scipy.stats import chi2
import unittest


class TestEverything(unittest.TestCase):

    def test_nist_approx(self):
        # values from:
        # https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Tolerance_Intervals_for_Normal_Data.pdf
        G = [0.9, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
             0.95]
        N = [26, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 5910, 866, 179]
        P = [0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        K = [1.6124, 1.7984, 1.7493, 1.7287, 1.7168, 1.7088, 1.7029, 1.6984,
             1.6948, 1.6703, 1.7138, 1.8084]
        for i, k in enumerate(K):
            n = N[i]
            x = np.random.random(n)*10
            xmu = np.mean(np.log(x))
            xstd = np.std(np.log(x), ddof=1)
            p = P[i]
            g = G[i]
            bound = lognormal(x, p, g)
            k_hat_l = (xmu - np.log(bound[0, 0]))/xstd
            k_hat_u = (np.log(bound[0, 1]) - xmu)/xstd
            self.assertTrue(np.isclose(k, k_hat_l, rtol=1e-4, atol=1e-5))
            self.assertTrue(np.isclose(k, k_hat_u, rtol=1e-4, atol=1e-5))

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        for m in M:
            for n in N:
                x = np.random.random((m, n))
                bounds = lognormal(x, 0.95, 0.95)
                _m, _ = bounds.shape
                self.assertTrue(_m == m)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            lognormal(x, 0.9, 0.9)


if __name__ == '__main__':
    unittest.main()
