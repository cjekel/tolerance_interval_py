# -- coding: utf-8 --
import numpy as np
from toleranceinterval.oneside import normal
import unittest


class TestEverything(unittest.TestCase):

    # Tabuluar values from Table XII in from:
    # Montgomery, D. C., & Runger, G. C. (2018). Chapter 8. Statistical
    # Intervals for a Single Sample. In Applied Statistics and Probability
    # for Engineers, 7th Edition.

    # Note there was a mistake in n=20, p=0.95, g=0.9....
    sample_sizes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50,
                            60, 70, 80, 90, 100])
    P = np.array([0.90, 0.95, 0.99, 0.90, 0.95, 0.99, 0.90, 0.95, 0.99])
    G = np.array([0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.99, 0.99, 0.99])
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

    K = K.reshape(sample_sizes.size, P.size)

    def test_upper_montgomery_bounds(self):
        for i, row in enumerate(self.K):
            n = self.sample_sizes[i]
            x = np.random.random(n)
            xmu = x.mean()
            xstd = x.std(ddof=1)
            for j, k in enumerate(row):
                p = self.P[j]
                g = self.G[j]
                bound = normal(x, p, g)
                k_hat = (bound - xmu)/xstd
                self.assertTrue(np.isclose(k, k_hat[0], rtol=1e-3, atol=1e-4))

    def test_lower_montgomery_bounds(self):
        for i, row in enumerate(self.K):
            n = self.sample_sizes[i]
            x = np.random.random(n)
            xmu = x.mean()
            xstd = x.std(ddof=1)
            for j, k in enumerate(row):
                p = 1.0 - self.P[j]
                g = self.G[j]
                bound = normal(x, p, g)
                k_hat = (xmu - bound)/xstd
                self.assertTrue(np.isclose(k, k_hat[0], rtol=1e-3, atol=1e-4))

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        for m in M:
            for n in N:
                x = np.random.random((m, n))
                bounds = normal(x, 0.1, 0.95)
                _m = bounds.size
                self.assertTrue(_m == m)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            normal(x, 0.1, 0.9)

    def test_lists(self):
        M = [3, 5]
        N = [5, 7]
        for m in M:
            for n in N:
                x = np.random.random((m, n))
                bounds = normal(list(x), 0.1, 0.95)
                _m = bounds.size
                self.assertTrue(_m == m)


if __name__ == '__main__':
    np.random.seed(121)
    unittest.main()
