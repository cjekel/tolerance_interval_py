# -- coding: utf-8 --
import numpy as np
from toleranceinterval.oneside import hanson_koopmans_cmh
import unittest


class TestEverything(unittest.TestCase):

    # B and A basis values from:
    # Volume 1: Guidelines for Characterization of Structural Materials.
    # (2017). In Composite Materials Handbook. SAE International.

    b_range = [35.177, 7.859, 4.505, 4.101, 3.064, 2.858, 2.382, 2.253, 2.137,
               1.897, 1.814, 1.738, 1.599, 1.540, 1.485, 1.434, 1.354, 1.311,
               1.253, 1.218, 1.184, 1.143, 1.114, 1.087, 1.060, 1.035, 1.010]
    n_range_b = list(range(2, 29))
    j_range = [2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10,
               10, 11, 11, 11, 11, 11, 12]
    a_range = [80.00380, 16.91220, 9.49579, 6.89049, 5.57681, 4.78352, 4.25011,
               3.86502, 3.57267, 3.34227, 3.15540, 3.00033, 2.86924, 2.75672,
               2.65889, 2.57290, 2.49660, 2.42833, 2.36683, 2.31106, 2.26020,
               2.21359, 2.17067, 2.13100, 2.09419, 2.05991, 2.02790, 1.99791,
               1.96975, 1.94324, 1.91822, 1.89457, 1.87215, 1.85088, 1.83065,
               1.81139, 1.79301, 1.77546, 1.75868, 1.74260, 1.72718, 1.71239,
               1.69817, 1.68449, 1.67132, 1.65862, 1.64638, 1.63456, 1.62313,
               1.60139, 1.58101, 1.56184, 1.54377, 1.52670, 1.51053, 1.49520,
               1.48063, 1.46675, 1.45352, 1.44089, 1.42881, 1.41724, 1.40614,
               1.39549, 1.38525, 1.37541, 1.36592, 1.35678, 1.34796, 1.33944,
               1.33120, 1.32324, 1.31553, 1.30806, 1.29036, 1.27392, 1.25859,
               1.24425, 1.23080, 1.21814, 1.20620, 1.19491, 1.18421, 1.17406,
               1.16440, 1.15519, 1.14640, 1.13801, 1.12997, 1.12226, 1.11486,
               1.10776, 1.10092, 1.09434, 1.08799, 1.08187, 1.07595, 1.07024,
               1.06471, 1.05935, 1.05417, 1.04914, 1.04426, 1.03952, 1.01773]
    n_range0 = np.arange(2, 51)
    n_range1 = np.arange(52, 102, 2)
    n_range2 = np.arange(105, 255, 5)
    n_range3 = [275]
    n_range = np.concatenate((n_range0, n_range1, n_range2, n_range3))

    def test_b_basis(self):
        p = 0.1
        g = 0.95
        for i, b in enumerate(self.b_range):
            j = self.j_range[i]-1
            n = self.n_range_b[i]
            x = np.random.random(n)
            x.sort()
            bound = hanson_koopmans_cmh(x, p, g, j=j)[0]
            b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
            self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_a_basis(self):
        p = 0.01
        g = 0.95
        for i, b in enumerate(self.a_range):
            n = self.n_range[i]
            j = n-1
            x = np.random.random(n)
            x.sort()
            bound = hanson_koopmans_cmh(x, p, g)[0]
            b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
            self.assertTrue(np.isclose(b, b_, rtol=1e-4, atol=1e-5))

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        J = [1, 2]
        for m in M:
            for n in N:
                for j in J:
                    x = np.random.random((m, n))
                    bounds = hanson_koopmans_cmh(x, 0.1, 0.95, j=j)
                    _m = bounds.size
                    self.assertTrue(_m == m)

    def test_value_error_shape(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            hanson_koopmans_cmh(x, 0.1, 0.9)

    def test_value_error_upper(self):
        with self.assertRaises(ValueError):
            x = np.random.random((10, 10))
            hanson_koopmans_cmh(x, 0.9, 0.95)

    def test_step_size(self):
        p = 0.1
        g = 0.95
        i = 0
        b = self.b_range[i]
        j = self.j_range[i]-1
        n = self.n_range_b[i]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans_cmh(x, p, g, j=j, step_size=1e-5)[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_new_raphson(self):
        p = 0.1
        g = 0.95
        i = 0
        b = self.b_range[i]
        j = self.j_range[i]-1
        n = self.n_range_b[i]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='newton-raphson')[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='newton-raphson',
                                    max_iter=50)[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='newton-raphson',
                                    tol=1e-6)[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_halley(self):
        p = 0.1
        g = 0.95
        i = 0
        b = self.b_range[i]
        j = self.j_range[i]-1
        n = self.n_range_b[i]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='halley')[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='halley',
                                    max_iter=50)[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans_cmh(x, p, g, j=j, method='halley',
                                    tol=1e-6)[0]
        b_ = np.log(bound / x[j]) / np.log(x[0]/x[j])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_fall_back(self):
        p = 0.01
        g = 0.95
        n = 300
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans_cmh(x, p, g)[0]
        self.assertTrue(np.isclose(bound, x[0]))


if __name__ == '__main__':
    np.random.seed(121)
    unittest.main()
