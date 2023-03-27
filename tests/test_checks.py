# -- coding: utf-8 --
import numpy as np
from toleranceinterval import checks
import unittest


class TestEverything(unittest.TestCase):

    def test_assert_2d_sort(self):
        for i in range(10):
            x = np.random.random(5)
            x = checks.numpy_array(x)
            x_sort = x.copy()
            x_sort.sort()
            x = checks.assert_2d_sort(x)
            for idx, x_new in enumerate(x[0]):
                # print(x_new, x_sort[idx])
                self.assertTrue(np.isclose(x_new, x_sort[idx]))


if __name__ == '__main__':
    unittest.main()
