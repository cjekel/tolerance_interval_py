# Author: Copyright (c) 2021 Jed Ludlow
# License: MIT License

"""
Test normal_factor against tables standard values from Meeker, Hahn, and Escobar.

Meeker, William Q.; Hahn, Gerald J.; Escobar, Luis A.. Statistical
Intervals: A Guide for Practitioners and Researchers (Wiley Series in
Probability and Statistics). Wiley. Kindle Edition.

Tables J.5a and J.5b provide tolerance factors for various combinations
of sample size, coverage, and confidence.

"""

import numpy as np
import toleranceinterval.twoside as ts
import unittest


class BaseTestMHE:

    class TestMeekerHahnEscobarJ5(unittest.TestCase):

        def test_tolerance_factor(self):
            for row_idx, row in enumerate(self.factor_g):
                for col_idx, g in enumerate(row):
                    k = ts.normal_factor(
                        self.sample_size[row_idx],
                        self.coverage[col_idx],
                        self.confidence[col_idx],
                        method='exact')
                    self.assertAlmostEqual(k, g, places=3)


class TestMeekerHahnEscobarJ5a(BaseTestMHE.TestMeekerHahnEscobarJ5):

    # Table J.5a from Meeker, William Q.; Hahn, Gerald J.; Escobar, Luis A..
    # Statistical Intervals: A Guide for Practitioners and Researchers
    # (Wiley Series in Probability and Statistics) (p. 535). Wiley.

    coverage = np.array([
        0.5, 0.5, 0.5, 0.5, 0.5,
        0.7, 0.7, 0.7, 0.7, 0.7,
        0.8, 0.8, 0.8, 0.8, 0.8,
    ])

    confidence = np.array([
        0.5, 0.8, 0.9, 0.95, 0.99,
        0.5, 0.8, 0.9, 0.95, 0.99,
        0.5, 0.8, 0.9, 0.95, 0.99,
    ])

    sample_size = np.array([
        2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        35, 40, 50, 60, 120, 240, 480, np.inf,
    ])

    factor_g = np.array([
        # n = 2
        1.243, 3.369, 6.808, 13.652, 68.316, 1.865, 5.023, 10.142,
        20.331, 101.732, 2.275, 6.110, 12.333, 24.722, 123.699,
        # n = 3
        0.942, 1.700, 2.492, 3.585, 8.122, 1.430, 2.562, 3.747,
        5.382, 12.181, 1.755, 3.134, 4.577, 6.572, 14.867,
        # n = 4
        0.852, 1.335, 1.766, 2.288, 4.028, 1.300, 2.026, 2.673,
        3.456, 6.073, 1.600, 2.486, 3.276, 4.233, 7.431,
        # n = 5
        0.808, 1.173, 1.473, 1.812, 2.824, 1.236, 1.788, 2.239,
        2.750, 4.274, 1.523, 2.198, 2.750, 3.375, 5.240,
        # n = 6
        0.782, 1.081, 1.314, 1.566, 2.270, 1.198, 1.651, 2.003,
        2.384, 3.446, 1.477, 2.034, 2.464, 2.930, 4.231,
        # n = 7
        0.764, 1.021, 1.213, 1.415, 1.954, 1.172, 1.562, 1.853,
        2.159, 2.973, 1.446, 1.925, 2.282, 2.657, 3.656,
        # n = 8
        0.752, 0.979, 1.143, 1.313, 1.750, 1.153, 1.499, 1.749,
        2.006, 2.668, 1.423, 1.849, 2.156, 2.472, 3.284,
        # n = 9
        0.742, 0.947, 1.092, 1.239, 1.608, 1.139, 1.451, 1.672,
        1.896, 2.455, 1.407, 1.791, 2.062, 2.337, 3.024,
        # n = 10
        0.735, 0.923, 1.053, 1.183, 1.503, 1.128, 1.414, 1.613,
        1.811, 2.297, 1.393, 1.746, 1.990, 2.234, 2.831,
        # n = 11
        0.729, 0.903, 1.021, 1.139, 1.422, 1.119, 1.385, 1.566,
        1.744, 2.175, 1.382, 1.710, 1.932, 2.152, 2.682,
        # n = 12
        0.724, 0.886, 0.996, 1.103, 1.357, 1.112, 1.360, 1.527,
        1.690, 2.078, 1.374, 1.680, 1.885, 2.086, 2.563,
        # n = 13
        0.720, 0.873, 0.974, 1.073, 1.305, 1.106, 1.339, 1.495,
        1.645, 1.999, 1.366, 1.654, 1.846, 2.031, 2.466,
        # n = 14
        0.717, 0.861, 0.956, 1.048, 1.261, 1.100, 1.322, 1.467,
        1.608, 1.933, 1.360, 1.633, 1.812, 1.985, 2.386,
        # n = 15
        0.714, 0.851, 0.941, 1.027, 1.224, 1.096, 1.307, 1.444,
        1.575, 1.877, 1.355, 1.614, 1.783, 1.945, 2.317,
        # n = 16
        0.711, 0.842, 0.927, 1.008, 1.193, 1.092, 1.293, 1.423,
        1.547, 1.829, 1.350, 1.598, 1.758, 1.911, 2.259,
        # n = 17
        0.709, 0.835, 0.915, 0.992, 1.165, 1.089, 1.282, 1.405,
        1.522, 1.788, 1.346, 1.584, 1.736, 1.881, 2.207,
        # n = 18
        0.707, 0.828, 0.905, 0.978, 1.141, 1.086, 1.271, 1.389,
        1.501, 1.751, 1.342, 1.571, 1.717, 1.854, 2.163,
        # n = 19
        0.705, 0.822, 0.895, 0.965, 1.120, 1.083, 1.262, 1.375,
        1.481, 1.719, 1.339, 1.559, 1.699, 1.830, 2.123,
        # n = 20
        0.704, 0.816, 0.887, 0.953, 1.101, 1.081, 1.253, 1.362,
        1.464, 1.690, 1.336, 1.549, 1.683, 1.809, 2.087,
        # n = 21
        0.702, 0.811, 0.879, 0.943, 1.084, 1.079, 1.246, 1.350,
        1.448, 1.664, 1.333, 1.540, 1.669, 1.789, 2.055,
        # n = 22
        0.701, 0.806, 0.872, 0.934, 1.068, 1.077, 1.239, 1.340,
        1.434, 1.640, 1.331, 1.531, 1.656, 1.772, 2.026,
        # n = 23
        0.700, 0.802, 0.866, 0.925, 1.054, 1.075, 1.232, 1.330,
        1.420, 1.619, 1.329, 1.523, 1.644, 1.755, 2.000,
        # n = 24
        0.699, 0.798, 0.860, 0.917, 1.042, 1.073, 1.226, 1.321,
        1.408, 1.599, 1.327, 1.516, 1.633, 1.741, 1.976,
        # n = 25
        0.698, 0.795, 0.855, 0.910, 1.030, 1.072, 1.221, 1.313,
        1.397, 1.581, 1.325, 1.509, 1.623, 1.727, 1.954,
        # n = 26
        0.697, 0.791, 0.850, 0.903, 1.019, 1.070, 1.216, 1.305,
        1.387, 1.565, 1.323, 1.503, 1.613, 1.714, 1.934,
        # n = 27
        0.696, 0.788, 0.845, 0.897, 1.009, 1.069, 1.211, 1.298,
        1.378, 1.550, 1.322, 1.497, 1.604, 1.703, 1.915,
        # n = 28
        0.695, 0.785, 0.841, 0.891, 1.000, 1.068, 1.207, 1.291,
        1.369, 1.535, 1.320, 1.492, 1.596, 1.692, 1.898,
        # n = 29
        0.694, 0.783, 0.837, 0.886, 0.991, 1.067, 1.202, 1.285,
        1.360, 1.522, 1.319, 1.487, 1.589, 1.682, 1.882,
        # n = 30
        0.694, 0.780, 0.833, 0.881, 0.983, 1.066, 1.199, 1.279,
        1.353, 1.510, 1.318, 1.482, 1.581, 1.672, 1.866,
        # n = 35
        0.691, 0.770, 0.817, 0.859, 0.950, 1.061, 1.182, 1.255,
        1.320, 1.459, 1.312, 1.462, 1.551, 1.632, 1.803,
        # n = 40
        0.689, 0.761, 0.805, 0.843, 0.925, 1.058, 1.170, 1.236,
        1.296, 1.420, 1.308, 1.446, 1.528, 1.602, 1.756,
        # n = 50
        0.686, 0.750, 0.787, 0.820, 0.889, 1.054, 1.152, 1.209,
        1.260, 1.365, 1.303, 1.424, 1.495, 1.558, 1.688,
        # n = 60
        0.684, 0.741, 0.775, 0.804, 0.864, 1.051, 1.139, 1.190,
        1.235, 1.327, 1.299, 1.408, 1.471, 1.527, 1.641,
        # n = 120
        0.679, 0.718, 0.740, 0.759, 0.797, 1.044, 1.104, 1.137,
        1.166, 1.225, 1.290, 1.365, 1.406, 1.442, 1.514,
        # n = 240
        0.677, 0.704, 0.719, 0.731, 0.756, 1.040, 1.082, 1.104,
        1.124, 1.162, 1.286, 1.337, 1.365, 1.390, 1.437,
        # n = 480
        0.676, 0.694, 0.705, 0.713, 0.730, 1.038, 1.067, 1.083,
        1.096, 1.122, 1.284, 1.320, 1.339, 1.355, 1.387,
        # n = infinity
        0.674, 0.674, 0.674, 0.674, 0.674, 1.036, 1.036, 1.036,
        1.036, 1.036, 1.282, 1.282, 1.282, 1.282, 1.282,
    ])

    factor_g = factor_g.reshape(sample_size.size, coverage.size)


class TestMeekerHahnEscobarJ5b(BaseTestMHE.TestMeekerHahnEscobarJ5):

    # Table J.5b from Meeker, William Q.; Hahn, Gerald J.; Escobar, Luis A..
    # Statistical Intervals: A Guide for Practitioners and Researchers
    # (Wiley Series in Probability and Statistics) (p. 535). Wiley.

    coverage = np.array([
        0.90, 0.90, 0.90, 0.90, 0.90,
        0.95, 0.95, 0.95, 0.95, 0.95,
        0.99, 0.99, 0.99, 0.99, 0.99,
    ])

    confidence = np.array([
        0.5, 0.8, 0.9, 0.95, 0.99,
        0.5, 0.8, 0.9, 0.95, 0.99,
        0.5, 0.8, 0.9, 0.95, 0.99,
    ])

    sample_size = np.array([
        2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        35, 40, 50, 60, 120, 240, 480, np.inf,
    ])

    factor_g = np.array([
        # n = 2
        2.869, 7.688, 15.512, 31.092, 155.569, 3.376, 9.032, 18.221,
        36.519, 182.720, 4.348, 11.613, 23.423, 46.944, 234.877,
        # n = 3
        2.229, 3.967, 5.788, 8.306, 18.782, 2.634, 4.679, 6.823, 9.789,
        22.131, 3.415, 6.051, 8.819, 12.647, 28.586,
        # n = 4
        2.039, 3.159, 4.157, 5.368, 9.416, 2.416, 3.736, 4.913, 6.341,
        11.118, 3.144, 4.850, 6.372, 8.221, 14.405,
        # n = 5
        1.945, 2.801, 3.499, 4.291, 6.655, 2.308, 3.318, 4.142, 5.077,
        7.870, 3.010, 4.318, 5.387, 6.598, 10.220,
        # n = 6
        1.888, 2.595, 3.141, 3.733, 5.383, 2.243, 3.078, 3.723, 4.422,
        6.373, 2.930, 4.013, 4.850, 5.758, 8.292,
        # n = 7
        1.850, 2.460, 2.913, 3.390, 4.658, 2.199, 2.920, 3.456, 4.020,
        5.520, 2.876, 3.813, 4.508, 5.241, 7.191,
        # n = 8
        1.823, 2.364, 2.754, 3.156, 4.189, 2.167, 2.808, 3.270, 3.746,
        4.968, 2.836, 3.670, 4.271, 4.889, 6.479,
        # n = 9
        1.802, 2.292, 2.637, 2.986, 3.860, 2.143, 2.723, 3.132, 3.546,
        4.581, 2.806, 3.562, 4.094, 4.633, 5.980,
        # n = 10
        1.785, 2.235, 2.546, 2.856, 3.617, 2.124, 2.657, 3.026, 3.393,
        4.294, 2.783, 3.478, 3.958, 4.437, 5.610,
        # n = 11
        1.772, 2.189, 2.473, 2.754, 3.429, 2.109, 2.604, 2.941, 3.273,
        4.073, 2.764, 3.410, 3.849, 4.282, 5.324,
        # n = 12
        1.761, 2.152, 2.414, 2.670, 3.279, 2.096, 2.560, 2.871, 3.175,
        3.896, 2.748, 3.353, 3.759, 4.156, 5.096,
        # n = 13
        1.752, 2.120, 2.364, 2.601, 3.156, 2.085, 2.522, 2.812, 3.093,
        3.751, 2.735, 3.306, 3.684, 4.051, 4.909,
        # n = 14
        1.744, 2.093, 2.322, 2.542, 3.054, 2.076, 2.490, 2.762, 3.024,
        3.631, 2.723, 3.265, 3.620, 3.962, 4.753,
        # n = 15
        1.737, 2.069, 2.285, 2.492, 2.967, 2.068, 2.463, 2.720, 2.965,
        3.529, 2.714, 3.229, 3.565, 3.885, 4.621,
        # n = 16
        1.731, 2.049, 2.254, 2.449, 2.893, 2.061, 2.439, 2.682, 2.913,
        3.441, 2.705, 3.198, 3.517, 3.819, 4.507,
        # n = 17
        1.726, 2.030, 2.226, 2.410, 2.828, 2.055, 2.417, 2.649, 2.868,
        3.364, 2.698, 3.171, 3.474, 3.761, 4.408,
        # n = 18
        1.721, 2.014, 2.201, 2.376, 2.771, 2.050, 2.398, 2.620, 2.828,
        3.297, 2.691, 3.146, 3.436, 3.709, 4.321,
        # n = 19
        1.717, 2.000, 2.178, 2.346, 2.720, 2.045, 2.381, 2.593, 2.793,
        3.237, 2.685, 3.124, 3.402, 3.663, 4.244,
        # n = 20
        1.714, 1.987, 2.158, 2.319, 2.675, 2.041, 2.365, 2.570, 2.760,
        3.184, 2.680, 3.104, 3.372, 3.621, 4.175,
        # n = 21
        1.710, 1.975, 2.140, 2.294, 2.635, 2.037, 2.351, 2.548, 2.731,
        3.136, 2.675, 3.086, 3.344, 3.583, 4.113,
        # n = 22
        1.707, 1.964, 2.123, 2.272, 2.598, 2.034, 2.338, 2.528, 2.705,
        3.092, 2.670, 3.070, 3.318, 3.549, 4.056,
        # n = 23
        1.705, 1.954, 2.108, 2.251, 2.564, 2.030, 2.327, 2.510, 2.681,
        3.053, 2.666, 3.054, 3.295, 3.518, 4.005,
        # n = 24
        1.702, 1.944, 2.094, 2.232, 2.534, 2.027, 2.316, 2.494, 2.658,
        3.017, 2.662, 3.040, 3.274, 3.489, 3.958,
        # n = 25
        1.700, 1.936, 2.081, 2.215, 2.506, 2.025, 2.306, 2.479, 2.638,
        2.984, 2.659, 3.027, 3.254, 3.462, 3.915,
        # n = 26
        1.698, 1.928, 2.069, 2.199, 2.480, 2.022, 2.296, 2.464, 2.619,
        2.953, 2.656, 3.015, 3.235, 3.437, 3.875,
        # n = 27
        1.696, 1.921, 2.058, 2.184, 2.456, 2.020, 2.288, 2.451, 2.601,
        2.925, 2.653, 3.004, 3.218, 3.415, 3.838,
        # n = 28
        1.694, 1.914, 2.048, 2.170, 2.434, 2.018, 2.279, 2.439, 2.585,
        2.898, 2.650, 2.993, 3.202, 3.393, 3.804,
        # n = 29
        1.692, 1.907, 2.038, 2.157, 2.413, 2.016, 2.272, 2.427, 2.569,
        2.874, 2.648, 2.983, 3.187, 3.373, 3.772,
        # n = 30
        1.691, 1.901, 2.029, 2.145, 2.394, 2.014, 2.265, 2.417, 2.555,
        2.851, 2.645, 2.974, 3.173, 3.355, 3.742,
        # n = 35
        1.684, 1.876, 1.991, 2.094, 2.314, 2.006, 2.234, 2.371, 2.495,
        2.756, 2.636, 2.935, 3.114, 3.276, 3.618,
        # n = 40
        1.679, 1.856, 1.961, 2.055, 2.253, 2.001, 2.211, 2.336, 2.448,
        2.684, 2.628, 2.905, 3.069, 3.216, 3.524,
        # n = 50
        1.672, 1.827, 1.918, 1.999, 2.166, 1.992, 2.177, 2.285, 2.382,
        2.580, 2.618, 2.861, 3.003, 3.129, 3.390,
        # n = 60
        1.668, 1.807, 1.888, 1.960, 2.106, 1.987, 2.154, 2.250, 2.335,
        2.509, 2.611, 2.830, 2.956, 3.068, 3.297,
        # n = 120
        1.656, 1.752, 1.805, 1.851, 1.943, 1.974, 2.087, 2.151, 2.206,
        2.315, 2.594, 2.743, 2.826, 2.899, 3.043,
        # n = 240
        1.651, 1.716, 1.753, 1.783, 1.844, 1.967, 2.045, 2.088, 2.125,
        2.197, 2.585, 2.688, 2.744, 2.793, 2.887,
        # n = 480
        1.648, 1.694, 1.718, 1.739, 1.780, 1.963, 2.018, 2.048, 2.073,
        2.121, 2.580, 2.652, 2.691, 2.724, 2.787,
        # n = infinity
        1.645, 1.645, 1.645, 1.645, 1.645, 1.960, 1.960, 1.960, 1.960,
        1.960, 2.576, 2.576, 2.576, 2.576, 2.576,
    ])

    factor_g = factor_g.reshape(sample_size.size, coverage.size)
