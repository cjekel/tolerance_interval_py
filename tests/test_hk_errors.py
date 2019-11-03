# -- coding: utf-8 --
from toleranceinterval.hk import HansonKoopmans
import unittest


class TestEverything(unittest.TestCase):

    def test_p_value_error(self):
        with self.assertRaises(ValueError):
            _ = HansonKoopmans(-.1, 0.9, 10, 8)

    def test_g_value_error(self):
        with self.assertRaises(ValueError):
            _ = HansonKoopmans(.1, -0.9, 10, 8)

    def test_j_value_error(self):
        with self.assertRaises(ValueError):
            _ = HansonKoopmans(.1, 0.9, 10, -200)


if __name__ == '__main__':
    unittest.main()
