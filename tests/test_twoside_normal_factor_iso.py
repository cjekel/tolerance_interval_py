# Author: Copyright (c) 2021 Jed Ludlow
# License: MIT License

"""
Test normla_factor against standard tables of tolerance factors
as published in ISO 16269-6:2014 Annex F.

A sampling of values from the tables is included here for brevity.

"""

import numpy as np
import toleranceinterval.twoside as ts
import unittest


def decimal_ceil(x, places):
    """
    Apply ceiling function at a decimal place.

    The tables of tolerance factors in ISO 16269-6:2014 provide
    the tolerance factors to a certain number of decimal places. The values
    at that final decimal place reflect the application of the ceiling function
    at that decimal place.

    """
    x *= 10 ** places
    x = np.ceil(x)
    x /= 10 ** places
    return x


class BaseTestIso:

    class TestIsoTableF(unittest.TestCase):

        def test_tolerance_factor(self):
            for row_idx, row in enumerate(self.factor_k5):
                for col_idx, k5 in enumerate(row):
                    k = ts.normal_factor(
                        self.sample_size[row_idx],
                        self.coverage,
                        self.confidence,
                        method='exact',
                        m=self.number_of_samples[col_idx])
                    k = decimal_ceil(k, places=4)
                    self.assertAlmostEqual(k, k5, places=4)


class TestIsoF1(BaseTestIso.TestIsoTableF):

    coverage = 0.90

    confidence = 0.90

    # This is n from the table.
    sample_size = np.array([
        2, 8, 16, 35, 100, 300, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 2
        15.5124, 6.0755, 4.5088, 3.8875, 3.5544,
        3.3461, 3.2032, 3.0989, 3.0193, 2.9565,
        # n = 8
        2.7542, 2.3600, 2.2244, 2.1530, 2.1081,
        2.0769, 2.0539, 2.0361, 2.0220, 2.0104,
        # n = 16
        2.2537, 2.0574, 1.9833, 1.9426, 1.9163,
        1.8977, 1.8837, 1.8727, 1.8638, 1.8564,
        # n = 35
        1.9906, 1.8843, 1.8417, 1.8176, 1.8017,
        1.7902, 1.7815, 1.7747, 1.7690, 1.7643,
        # n = 100
        1.8232, 1.7697, 1.7473, 1.7343, 1.7256,
        1.7193, 1.7144, 1.7105, 1.7073, 1.7047,
        # n = 300
        1.7401, 1.7118, 1.6997, 1.6925, 1.6877,
        1.6842, 1.6815, 1.6793, 1.6775, 1.6760,
        # n = 1000
        1.6947, 1.6800, 1.6736, 1.6698, 1.6672,
        1.6653, 1.6639, 1.6627, 1.6617, 1.6609,
        # n = infinity
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF2(BaseTestIso.TestIsoTableF):

    coverage = 0.95

    confidence = 0.90

    # This is n from the table.
    sample_size = np.array([
        3, 9, 15, 30, 90, 150, 400, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 3
        6.8233, 4.3320, 3.7087, 3.4207, 3.2528,
        3.1420, 3.0630, 3.0038, 2.9575, 2.9205,
        # n = 9
        3.1323, 2.7216, 2.5773, 2.5006, 2.4521,
        2.4182, 2.3931, 2.3737, 2.3581, 2.3454,
        # n = 15
        2.7196, 2.4718, 2.3789, 2.3280, 2.2951,
        2.2719, 2.2545, 2.2408, 2.2298, 2.2206,
        # n = 30
        2.4166, 2.2749, 2.2187, 2.1870, 2.1662,
        2.1513, 2.1399, 2.1309, 2.1236, 2.1175,
        # n = 90
        2.1862, 2.1182, 2.0898, 2.0733, 2.0624,
        2.0544, 2.0482, 2.0433, 2.0393, 2.0360,
        # n = 150
        2.1276, 2.0775, 2.0563, 2.0439, 2.0356,
        2.0296, 2.0249, 2.0212, 2.0181, 2.0155,
        # n = 400
        2.0569, 2.0282, 2.0158, 2.0085, 2.0035,
        1.9999, 1.9971, 1.9949, 1.9930, 1.9915,
        # n = 1000
        2.0193, 2.0018, 1.9942, 1.9897, 1.9866,
        1.9844, 1.9826, 1.9812, 1.9800, 1.9791,
        # n = infinity
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF3(BaseTestIso.TestIsoTableF):

    coverage = 0.99

    confidence = 0.90

    # This is n from the table.
    sample_size = np.array([
        4, 8, 17, 28, 100, 300, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 4
        6.3722, 4.6643, 4.1701, 3.9277, 3.7814,
        3.6825, 3.6108, 3.5562, 3.5131, 3.4782,
        # n = 8
        4.2707, 3.6541, 3.4408, 3.3281, 3.2572,
        3.2078, 3.1712, 3.1428, 3.1202, 3.1016,
        # n = 17
        3.4741, 3.1819, 3.0708, 3.0095, 2.9698,
        2.9416, 2.9204, 2.9037, 2.8902, 2.8791,
        # n = 28
        3.2023, 3.0062, 2.9286, 2.8850, 2.8564,
        2.8358, 2.8203, 2.8080, 2.7980, 2.7896,
        # n = 100
        2.8548, 2.7710, 2.7358, 2.7155, 2.7018,
        2.6919, 2.6843, 2.6782, 2.6732, 2.6690,
        # n = 300
        2.7249, 2.6806, 2.6616, 2.6504, 2.6429,
        2.6374, 2.6331, 2.6297, 2.6269, 2.6245,
        # n = 1000
        2.6538, 2.6308, 2.6208, 2.6148, 2.6108,
        2.6079, 2.6056, 2.6037, 2.6022, 2.6009,
        # n = infinity
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF4(BaseTestIso.TestIsoTableF):

    coverage = 0.90

    confidence = 0.95

    # This is n from the table.
    sample_size = np.array([
        2, 8, 16, 35, 150, 500, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 2
        31.0923, 8.7252, 5.8380, 4.7912, 4.2571,
        3.9341, 3.7179, 3.5630, 3.4468, 3.3565,
        # n = 8
        3.1561, 2.5818, 2.3937, 2.2974, 2.2381,
        2.1978, 2.1685, 2.1463, 2.1289, 2.1149,
        # n = 16
        2.4486, 2.1771, 2.0777, 2.0241, 1.9899,
        1.9661, 1.9483, 1.9346, 1.9237, 1.9147,
        # n = 35
        2.0943, 1.9515, 1.8953, 1.8638, 1.8432,
        1.8285, 1.8174, 1.8087, 1.8016, 1.7957,
        # n = 150
        1.8260, 1.7710, 1.7478, 1.7344, 1.7254,
        1.7188, 1.7137, 1.7097, 1.7064, 1.7036,
        # n = 500
        1.7374, 1.7098, 1.6979, 1.6908, 1.6861,
        1.6826, 1.6799, 1.6777, 1.6760, 1.6744,
        # n = 1000
        1.7088, 1.6898, 1.6816, 1.6767, 1.6734,
        1.6709, 1.6690, 1.6675, 1.6663, 1.6652,
        # n = infinity
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF5(BaseTestIso.TestIsoTableF):

    coverage = 0.95

    confidence = 0.95

    # This is n from the table.
    sample_size = np.array([
        5, 10, 26, 90, 200, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 5
        5.0769, 3.6939, 3.2936, 3.0986, 2.9820,
        2.9041, 2.8482, 2.8062, 2.7734, 2.7472,
        # n = 10
        3.3935, 2.8700, 2.6904, 2.5964, 2.5377,
        2.4973, 2.4677, 2.4450, 2.4271, 2.4125,
        # n = 26
        2.6188, 2.4051, 2.3227, 2.2771, 2.2476,
        2.2266, 2.2108, 2.1985, 2.1886, 2.1803,
        # n = 90
        2.2519, 2.1622, 2.1251, 2.1037, 2.0895,
        2.0792, 2.0713, 2.0650, 2.0598, 2.0555,
        # n = 200
        2.1430, 2.0877, 2.0642, 2.0505, 2.0413,
        2.0346, 2.0294, 2.0253, 2.0219, 2.0190,
        # n = 1000
        2.0362, 2.0135, 2.0037, 1.9979, 1.9939,
        1.9910, 1.9888, 1.9870, 1.9855, 1.9842,
        # n = infinity
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF6(BaseTestIso.TestIsoTableF):

    coverage = 0.99

    confidence = 0.95

    # This is n from the table.
    sample_size = np.array([
        3, 9, 17, 35, 100, 500, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 3
        12.6472, 6.8474, 5.5623, 4.9943, 4.6711,
        4.4612, 4.3133, 4.2032, 4.1180, 4.0500,
        # n = 9
        4.6329, 3.8544, 3.5909, 3.4534, 3.3677,
        3.3085, 3.2651, 3.2317, 3.2052, 3.1837,
        # n = 17
        3.7606, 3.3572, 3.2077, 3.1264, 3.0743,
        3.0377, 3.0104, 2.9892, 2.9722, 2.9582,
        # n = 35
        3.2762, 3.0522, 2.9638, 2.9143, 2.8818,
        2.8586, 2.8411, 2.8273, 2.8161, 2.8068,
        # n = 100
        2.9356, 2.8253, 2.7794, 2.7529, 2.7352,
        2.7224, 2.7126, 2.7048, 2.6984, 2.6930,
        # n = 500
        2.7208, 2.6775, 2.6588, 2.6478, 2.6403,
        2.6349, 2.6307, 2.6273, 2.6245, 2.6221,
        # n = infinity
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF7(BaseTestIso.TestIsoTableF):

    coverage = 0.90

    confidence = 0.99

    # This is n from the table.
    sample_size = np.array([
        4, 10, 22, 80, 200, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 4
        9.4162, 4.9212, 3.9582, 3.5449, 3.3166,
        3.1727, 3.0742, 3.0028, 2.9489, 2.9068,
        # n = 10
        3.6167, 2.8193, 2.5709, 2.4481, 2.3748,
        2.3265, 2.2923, 2.2671, 2.2477, 2.2324,
        # n = 22
        2.5979, 2.2631, 2.1429, 2.0791, 2.0393,
        2.0120, 1.9921, 1.9770, 1.9652, 1.9558,
        # n = 80
        2.0282, 1.9056, 1.8562, 1.8281, 1.8097,
        1.7964, 1.7864, 1.7785, 1.7721, 1.7668,
        # n = 200
        1.8657, 1.7973, 1.7686, 1.7520, 1.7409,
        1.7328, 1.7266, 1.7216, 1.7176, 1.7142,
        # n = 1000
        1.7359, 1.7086, 1.6967, 1.6897, 1.6850,
        1.6815, 1.6788, 1.6767, 1.6749, 1.6734,
        # n = infinity
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
        1.6449, 1.6449, 1.6449, 1.6449, 1.6449,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF8(BaseTestIso.TestIsoTableF):

    coverage = 0.95

    confidence = 0.99

    # This is n from the table.
    sample_size = np.array([
        2, 9, 17, 40, 150, 500, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 2
        182.7201, 23.1159, 11.9855, 8.7010, 7.1975,
        6.3481, 5.8059, 5.4311, 5.1573, 4.9489,
        # n = 9
        4.5810, 3.4807, 3.1443, 2.9784, 2.8793,
        2.8136, 2.7670, 2.7324, 2.7057, 2.6846,
        # n = 17
        3.3641, 2.8501, 2.6716, 2.5784, 2.5207,
        2.4814, 2.4529, 2.4314, 2.4147, 2.4013,
        # n = 40
        2.6836, 2.4425, 2.3498, 2.2987, 2.2658,
        2.2427, 2.2254, 2.2120, 2.2013, 2.1926,
        # n = 150
        2.2712, 2.1740, 2.1336, 2.1103, 2.0948,
        2.0835, 2.0749, 2.0681, 2.0625, 2.0578,
        # n = 500
        2.1175, 2.0697, 2.0492, 2.0372, 2.0291,
        2.0231, 2.0185, 2.0149, 2.0118, 2.0093,
        # n = infinity
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
        1.9600, 1.9600, 1.9600, 1.9600, 1.9600,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)


class TestIsoF9(BaseTestIso.TestIsoTableF):

    coverage = 0.99

    confidence = 0.99

    # This is n from the table.
    sample_size = np.array([
        3, 7, 15, 28, 70, 200, 1000, np.inf,
    ])

    # This is m from the table.
    number_of_samples = np.arange(1, 11)

    factor_k5 = np.array([
        # n = 3
        28.5857, 10.6204, 7.6599, 6.4888, 5.8628,
        5.4728, 5.2065, 5.0131, 4.8663, 4.7512,
        # n = 7
        7.1908, 5.0656, 4.4559, 4.1605, 3.9847,
        3.8678, 3.7844, 3.7220, 3.6736, 3.6350,
        # n = 15
        4.6212, 3.8478, 3.5825, 3.4441, 3.3581,
        3.2992, 3.2564, 3.2238, 3.1983, 3.1777,
        # n = 28
        3.8042, 3.3792, 3.2209, 3.1350, 3.0801,
        3.0418, 3.0135, 2.9916, 2.9742, 2.9600,
        # n = 70
        3.2284, 3.0179, 2.9334, 2.8857, 2.8544,
        2.8319, 2.8150, 2.8016, 2.7908, 2.7818,
        # n = 200
        2.9215, 2.8144, 2.7695, 2.7434, 2.7260,
        2.7133, 2.7036, 2.6958, 2.6894, 2.6841,
        # n = 1000
        2.7184, 2.6756, 2.6570, 2.6461, 2.6387,
        2.6332, 2.6290, 2.6257, 2.6229, 2.6205,
        # n = infinity
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
        2.5759, 2.5759, 2.5759, 2.5759, 2.5759,
    ])

    factor_k5 = factor_k5.reshape(sample_size.size, number_of_samples.size)
