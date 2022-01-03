# -- coding: utf-8 --
import numpy as np
from toleranceinterval.oneside import non_parametric
import unittest


class TestEverything(unittest.TestCase):

    # Tabuluar values from Table J11 from:
    # Meeker, W.Q., Hahn, G.J. and Escobar, L.A., 2017. Statistical intervals:
    # A guide for practitioners and researchers (Vol. 541). John Wiley & Sons.

    sample_sizes = np.array([10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200,
                             300, 400, 500, 600, 800, 1000])
    P = np.array([0.75, 0.75, 0.75, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.99,
                  0.99, 0.99])
    G = np.array([0.90, 0.95, 0.99, 0.90, 0.95, 0.99, 0.90, 0.95, 0.99, 0.90,
                  0.95, 0.99])
    K = np.array([1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, 2, 1, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, 2,
                  1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, 4, 3, 2, 1, np.nan, np.nan, np.nan, np.nan,
                  np.nan, np.nan, np.nan, np.nan, 5, 4, 2, 1, 1, np.nan,
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 6, 5, 3, 1,
                  1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7,
                  6, 4, 2, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  np.nan, 9, 8, 6, 2, 2, 1, 1, np.nan, np.nan, np.nan, np.nan,
                  np.nan, 11, 10, 8, 3, 2, 1, 1, 1, np.nan, np.nan, np.nan,
                  np.nan, 15, 14, 11, 5, 4, 2, 2, 1, np.nan, np.nan, np.nan,
                  np.nan, 20, 18, 15, 6, 5, 4, 2, 2, 1, np.nan, np.nan, np.nan,
                  42, 40, 36, 15, 13, 11, 6, 5, 4, np.nan, np.nan, np.nan, 65,
                  63, 58, 23, 22, 19, 10, 9, 7, 1, 1, np.nan, 89, 86, 80, 32,
                  30, 27, 15, 13, 11, 2, 1, np.nan, 113, 109, 103, 41, 39, 35,
                  19, 17, 14, 2, 2, 1, 136, 133, 126, 51, 48, 44, 23, 21, 18,
                  3, 2, 1, 184, 180, 172, 69, 66, 61, 32, 30, 26, 5, 4, 2, 233,
                  228, 219, 88, 85, 79, 41, 39, 35, 6, 5, 3]) - 1.
    K = K.reshape(sample_sizes.size, P.size)

    def test_upper_table_bounds(self):
        for i, row in enumerate(self.K):
            n = self.sample_sizes[i]
            x = np.arange(n)
            for j, k in enumerate(row):
                k = n - k - 1
                p = self.P[j]
                g = self.G[j]
                bound = non_parametric(x, p, g)[0]
                if np.isnan(k) and np.isnan(bound):
                    self.assertTrue(True)
                else:
                    self.assertEqual(k, bound)

    def test_lower_table_bounds(self):
        for i, row in enumerate(self.K):
            n = self.sample_sizes[i]
            x = np.arange(n)
            for j, k in enumerate(row):
                p = 1.0 - self.P[j]
                g = self.G[j]
                bound = non_parametric(x, p, g)[0]
                if np.isnan(k) and np.isnan(bound):
                    self.assertTrue(True)
                else:
                    self.assertEqual(k, bound)

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        for m in M:
            for n in N:
                x = np.random.random((m, n))
                bounds = non_parametric(x, 0.1, 0.95)
                _m = bounds.size
                self.assertTrue(_m == m)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            non_parametric(x, 0.1, 0.9)


if __name__ == '__main__':
    np.random.seed(121)
    unittest.main()
