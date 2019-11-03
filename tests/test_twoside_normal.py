# -- coding: utf-8 --
import numpy as np
from toleranceinterval.twoside import normal
# from scipy.stats import chi2
import unittest


class TestEverything(unittest.TestCase):

    # Tabuluar values from Table XII in from:
    # Montgomery, D. C., & Runger, G. C. (2018). Chapter 8. Statistical
    # Intervals for a Single Sample. In Applied Statistics and Probability
    # for Engineers, 7th Edition.

    # Control",, Griffin's Statistical Monographs, Number 37, London.
    # sample_sizes = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    #                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40,
    # 50,
    #                         60, 70, 80, 90, 100])
    # P = np.array([0.90, 0.95, 0.99, 0.90, 0.95, 0.99, 0.90, 0.95, 0.99])
    # G = np.array([0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.99, 0.99, 0.99])
    # K = np.array([15.978, 18.800, 24.167, 32.019, 37.674, 48.430, 160.193,
    #               188.491, 242.300, 5.847, 6.919, 8.974, 8.380, 9.916,
    # 12.861,
    #               18.930, 22.401, 29.055, 4.166, 4.943, 6.440, 5.369, 6.370,
    #               8.299, 9.398, 11.150, 14.527, 3.949, 4.152, 5.423, 4.275,
    #               5.079, 6.634, 6.612, 7.855, 10.260, 3.131, 3.723, 4.870,
    #               3.712, 4.414, 5.775, 5.337, 6.345, 8.301, 2.902, 3.452,
    #               4.521, 3.369, 4.007, 5.248, 4.613, 5.488, 7.187, 2.743,
    #               3.264, 4.278, 3.136, 3.732, 4.891, 4.147, 4.936, 6.468,
    #               2.626, 3.125, 4.098, 2.967, 3.532, 4.631, 3.822, 4.550,
    #               5.966, 2.535, 3.018, 3.959, 2.839, 3.379, 4.433, 3.582,
    #               4.265, 5.594, 2.463, 2.933, 3.849, 2.737, 3.259, 4.277,
    #               3.397, 4.045, 5.308, 2.404, 2.863, 3.758, 2.655, 3.162,
    #               4.150, 3.250, 3.870, 5.079, 2.355, 2.805, 3.682, 2.587,
    #               3.081, 4.044, 3.130, 3.727, 4.893, 2.314, 2.756, 3.618,
    #               2.529, 3.012, 3.955, 3.029, 3.608, 4.737, 2.278, 2.713,
    #               3.562, 2.480, 2.954, 3.878, 2.945, 3.507, 4.605, 2.246,
    #               2.676, 3.514, 2.437, 2.903, 3.812, 2.872, 3.421, 4.492,
    #               2.219, 2.643, 3.471, 2.400, 2.858, 3.754, 2.808, 3.345,
    #               4.393, 2.194, 2.614, 3.433, 2.366, 2.819, 3.702, 2.753,
    #               3.279, 4.307, 2.172, 2.588, 3.399, 2.337, 2.784, 3.656,
    #               2.703, 3.221, 4.230, 2.152, 2.564, 3.368, 2.310, 2.752,
    #               3.615, 2.659, 3.168, 4.161, 2.135, 2.543, 3.340, 2.286,
    #               2.723, 3.577, 2.620, 3.121, 4.100, 2.118, 2.524, 3.315,
    #               2.264, 2.697, 3.543, 2.584, 3.078, 4.044, 2.103, 2.506,
    #               3.292, 2.244, 2.673, 3.512, 2.551, 3.040, 3.993, 2.089,
    #               2.489, 3.270, 2.225, 2.651, 3.483, 2.522, 3.004, 3.947,
    #               2.077, 2.474, 3.251, 2.208, 2.631, 3.457, 2.494, 2.972,
    #               3.904, 2.025, 2.413, 3.170, 2.140, 2.529, 3.350, 2.385,
    #               2.841, 3.733, 1.959, 2.334, 3.066, 2.052, 2.445, 3.213,
    #               2.247, 2.677, 3.518, 1.916, 2.284, 3.001, 1.996, 2.379,
    #               3.126, 2.162, 2.576, 3.385, 1.887, 2.248, 2.955, 1.958,
    #               2.333, 3.066, 2.103, 2.506, 3.293, 1.865, 2.222, 2.920,
    #               1.929, 2.299, 3.021, 2.060, 2.454, 3.225, 1.848, 2.202,
    #               2.894, 1.907, 2.272, 2.986, 2.026, 2.414, 3.173, 1.834,
    #               2.185, 2.872, 1.889, 2.251, 2.958, 1.999, 2.382, 3.130,
    #               1.822, 2.172, 2.854, 1.874, 2.233, 2.934, 1.977, 2.355,
    #               3.096])

    # K = K.reshape(sample_sizes.size, P.size)

    # def test_montgomery_bounds(self):
    #     for i, row in enumerate(self.K):
    #         n = self.sample_sizes[i]
    #         x = np.random.random(n)*10
    #         xmu = x.mean()
    #         xstd = x.std(ddof=1)

    #         for j, k in enumerate(row):
    #             p = self.P[j]
    #             g = self.G[j]
    #             chi2v = chi2.ppf(1-g, df=n-1.0)
    #             w = np.sqrt(1.0 + ((n-3.0-chi2v)/(2.0*(n+1.0)**2)))
    #             k = k*w
    #             bound = normal(x, p, g)
    #             k_hat_l = (xmu - bound[0, 0])/xstd
    #             k_hat_u = (bound[0, 1] - xmu)/xstd
    #             print(k, k_hat_l, k_hat_u, bound)
    #             self.assertTrue(np.isclose(k, k_hat_l, rtol=1e-3, atol=1e-4))
    #             self.assertTrue(np.isclose(k, k_hat_u, rtol=1e-3, atol=1e-4))

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
            xmu = x.mean()
            xstd = x.std(ddof=1)
            p = P[i]
            g = G[i]
            bound = normal(x, p, g)
            k_hat_l = (xmu - bound[0, 0])/xstd
            k_hat_u = (bound[0, 1] - xmu)/xstd
            self.assertTrue(np.isclose(k, k_hat_l, rtol=1e-4, atol=1e-5))
            self.assertTrue(np.isclose(k, k_hat_u, rtol=1e-4, atol=1e-5))

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
