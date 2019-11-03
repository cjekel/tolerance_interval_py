# -- coding: utf-8 --
import numpy as np
from toleranceinterval.oneside import hanson_koopmans
import unittest


class TestEverything(unittest.TestCase):

    # Values from:
    # Hanson, D. L., & Koopmans, L. H. (1964). Tolerance Limits for
    # the Class of Distributions with Increasing Hazard Rates. Ann. Math.
    # Statist., 35(4), 1561-1570. https://doi.org/10.1214/aoms/1177700380
    #
    # data[:, [n, p, g, b]]

    data = np.array([[2, 0.25, 0.9, 8.618],
                     [2, 0.25, 0.95, 17.80],
                     [2, 0.25, 0.99, 91.21],
                     [3, 0.25, 0.90, 5.898],
                     [3, 0.25, 0.95, 12.27],
                     [3, 0.25, 0.99, 63.17],
                     [4, 0.25, 0.90, 4.116],
                     [4, 0.25, 0.95, 8.638],
                     [4, 0.25, 0.99, 44.78],
                     [5, 0.25, 0.90, 2.898],
                     [5, 0.25, 0.95, 6.154],
                     [5, 0.25, 0.99, 32.17],
                     [6, 0.25, 0.90, 2.044],
                     [6, 0.25, 0.95, 4.411],
                     [6, 0.25, 0.99, 23.31],
                     [7, 0.25, 0.90, 1.437],
                     [7, 0.25, 0.95, 3.169],
                     [7, 0.25, 0.99, 16.98],
                     [8, 0.25, 0.90, 1.001],
                     [8, 0.25, 0.95, 2.275],
                     [8, 0.25, 0.99, 12.42],
                     [9, 0.25, 0.95, 1.627],
                     [9, 0.25, 0.99, 9.100],
                     [2, 0.10, 0.90, 17.09],
                     [2, 0.10, 0.95, 35.18],
                     [2, 0.10, 0.99, 179.8],
                     [3, 0.10, 0.90, 13.98],
                     [3, 0.10, 0.95, 28.82],
                     [3, 0.10, 0.99, 147.5],
                     [4, 0.10, 0.90, 11.70],
                     [4, 0.10, 0.95, 24.17],
                     [4, 0.10, 0.99, 123.9],
                     [5, 0.10, 0.90, 9.931],
                     [5, 0.10, 0.95, 20.57],
                     [5, 0.10, 0.99, 105.6],
                     [6, 0.10, 0.90, 8.512],
                     [6, 0.10, 0.95, 17.67],
                     [6, 0.10, 0.99, 90.90],
                     [7, 0.10, 0.90, 7.344],
                     [7, 0.10, 0.95, 15.29],
                     [7, 0.10, 0.99, 78.80],
                     [8, 0.10, 0.90, 6.368],
                     [8, 0.10, 0.95, 13.30],
                     [8, 0.10, 0.99, 68.68],
                     [9, 0.10, 0.90, 5.541],
                     [9, 0.10, 0.95, 11.61],
                     [9, 0.10, 0.99, 60.10],
                     [2, 0.05, 0.90, 23.65],
                     [2, 0.05, 0.95, 48.63],
                     [2, 0.05, 0.99, 248.4],
                     [3, 0.05, 0.90, 20.48],
                     [3, 0.05, 0.95, 42.15],
                     [3, 0.05, 0.99, 215.4],
                     [4, 0.05, 0.90, 18.12],
                     [4, 0.05, 0.95, 37.32],
                     [4, 0.05, 0.99, 190.9],
                     [5, 0.05, 0.90, 16.24],
                     [5, 0.05, 0.95, 33.49],
                     [5, 0.05, 0.99, 171.4],
                     [6, 0.05, 0.90, 14.70],
                     [6, 0.05, 0.95, 30.33],
                     [6, 0.05, 0.99, 155.4],
                     [7, 0.05, 0.90, 13.39],
                     [7, 0.05, 0.95, 27.66],
                     [7, 0.05, 0.99, 141.8],
                     [8, 0.05, 0.90, 12.26],
                     [8, 0.05, 0.95, 25.35],
                     [8, 0.05, 0.99, 130.0],
                     [9, 0.05, 0.90, 11.27],
                     [9, 0.05, 0.95, 23.33],
                     [9, 0.05, 0.99, 119.8],
                     [20, 0.05, 0.90, 5.077],
                     [20, 0.05, 0.95, 10.68],
                     [20, 0.05, 0.99, 55.47]])

    def test_upper_table_bounds(self):
        j = 1
        for i, row in enumerate(self.data):
            n = int(row[0])
            p = 1.0-row[1]
            g = row[2]
            b = row[3]
            x = np.random.random(n) + 1000.
            x.sort()
            bound = hanson_koopmans(x, p, g, j=1)[0]
            b_ = (bound - x[n-j-1]) / (x[-1] - x[n-j-1])
            self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_lower_table_bounds(self):
        j = 1
        for i, row in enumerate(self.data):
            n = int(row[0])
            p = row[1]
            g = row[2]
            b = row[3]
            x = np.random.random(n) + 1000.
            x.sort()
            bound = hanson_koopmans(x, p, g, j=1)[0]
            b_ = (x[j] - bound) / (x[j] - x[0])
            self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_random_shapes(self):
        M = [3, 10, 20]
        N = [5, 10, 20]
        J = [1, 2]
        for m in M:
            for n in N:
                for j in J:
                    x = np.random.random((m, n))
                    bounds = hanson_koopmans(x, 0.1, 0.95, j=j)
                    _m = bounds.size
                    self.assertTrue(_m == m)

    def test_value_error(self):
        with self.assertRaises(ValueError):
            x = np.random.random((1, 2, 4, 3))
            hanson_koopmans(x, 0.1, 0.9)

    def test_step_size(self):
        i = 0
        row = self.data[i]
        n = int(row[0])
        j = n-1
        p = row[1]
        g = row[2]
        b = row[3]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans(x, p, g, step_size=1e-6)[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_new_raphson(self):
        i = 0
        row = self.data[i]
        n = int(row[0])
        j = n-1
        p = row[1]
        g = row[2]
        b = row[3]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans(x, p, g, method='newton-raphson')[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans(x, p, g, method='newton-raphson',
                                max_iter=50)[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans(x, p, g, method='newton-raphson',
                                tol=1e-6)[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_halley(self):
        i = 0
        row = self.data[i]
        n = int(row[0])
        j = n-1
        p = row[1]
        g = row[2]
        b = row[3]
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans(x, p, g, method='halley')[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans(x, p, g, method='halley', max_iter=50)[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))
        bound = hanson_koopmans(x, p, g, method='halley', tol=1e-6)[0]
        b_ = (x[j] - bound) / (x[j] - x[0])
        self.assertTrue(np.isclose(b, b_, rtol=1e-3, atol=1e-4))

    def test_fall_back(self):
        p = 0.01
        g = 0.95
        n = 300
        x = np.random.random(n)
        x.sort()
        bound = hanson_koopmans(x, p, g)[0]
        self.assertTrue(np.isclose(bound, x[0]))


if __name__ == '__main__':
    np.random.seed(121)
    unittest.main()
