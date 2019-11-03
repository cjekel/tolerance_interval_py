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
    K = np.array([10.253, 13.090, 18.500, 20.581, 26.260, 37.094, 103.029,
                  131.426, 185.617, 4.258, 5.311, 7.340, 6.155, 7.656, 10.553,
                  13.995, 17.370, 23.896, 3.188, 3.957, 5.438, 4.162, 5.144,
                  7.042, 7.380, 9.083, 12.387, 2.742, 3.400, 4.666, 3.407,
                  4.203, 5.741, 5.362, 6.578, 8.939, 2.494, 3.092, 4.243,
                  3.006, 3.708, 5.062, 4.411, 5.406, 7.335, 2.333, 2.894,
                  3.972, 2.755, 3.399, 4.642, 3.859, 4.728, 6.412, 2.219,
                  2.754, 3.783, 2.582, 3.187, 4.354, 3.497, 4.285, 5.812,
                  2.133, 2.650, 3.641, 2.454, 3.031, 4.143, 3.240, 3.972,
                  5.389, 2.066, 2.568, 3.532, 2.355, 2.911, 3.981, 3.048,
                  3.738, 5.074, 2.011, 2.503, 3.443, 2.275, 2.815, 3.852,
                  2.898, 3.556, 4.829, 1.966, 2.448, 3.371, 2.210, 2.736,
                  3.747, 2.777, 3.410, 4.633, 1.928, 2.402, 3.309, 2.155,
                  2.671, 3.659, 2.677, 3.290, 4.472, 1.895, 2.363, 3.257,
                  2.109, 2.614, 3.585, 2.593, 3.189, 4.337, 1.867, 2.329,
                  3.212, 2.068, 2.566, 3.520, 2.521, 3.102, 4.222, 1.842,
                  2.299, 3.172, 2.033, 2.524, 3.464, 2.459, 3.028, 4.123,
                  1.819, 2.272, 3.137, 2.002, 2.486, 3.414, 2.405, 2.963,
                  4.037, 1.800, 2.249, 3.105, 1.974, 2.453, 3.370, 2.357,
                  2.905, 3.960, 1.782, 2.227, 3.077, 1.949, 2.423, 3.331,
                  2.314, 2.854, 3.892, 1.765, 2.208, 3.052, 1.926, 2.396,
                  3.295, 2.276, 2.808, 3.832, 1.750, 2.190, 3.028, 1.905,
                  2.371, 3.263, 2.241, 2.766, 3.777, 1.737, 2.174, 3.007,
                  1.886, 2.349, 3.233, 2.209, 2.729, 3.727, 1.724, 2.159,
                  2.987, 1.869, 2.328, 3.206, 2.180, 2.694, 3.681, 1.712,
                  2.145, 2.969, 1.853, 2.309, 3.181, 2.154, 2.662, 3.640,
                  1.702, 2.132, 2.952, 1.838, 2.292, 3.158, 2.129, 2.633,
                  3.601, 1.657, 2.080, 2.884, 1.777, 2.220, 3.064, 2.030,
                  2.515, 3.447, 1.598, 2.010, 2.793, 1.697, 2.125, 2.941,
                  1.902, 2.364, 3.249, 1.559, 1.965, 2.735, 1.646, 2.065,
                  2.862, 1.821, 2.269, 3.125, 1.532, 1.933, 2.694, 1.609,
                  2.022, 2.807, 1.764, 2.202, 3.038, 1.511, 1.909, 2.662,
                  1.581, 1.990, 2.765, 1.722, 2.153, 2.974, 1.495, 1.890,
                  2.638, 1.559, 1.964, 2.733, 1.688, 2.114, 2.924, 1.481,
                  1.874, 2.618, 1.542, 1.944, 2.706, 1.661, 2.082, 2.883,
                  1.470, 1.861, 2.601, 1.527, 1.927, 2.684, 1.639, 2.056,
                  2.850])

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
