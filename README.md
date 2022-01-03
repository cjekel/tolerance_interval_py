# About

### toleranceinterval

A small Python library for one-sided tolerance bounds and two-sided tolerance intervals.

[![Build Status](https://travis-ci.com/cjekel/tolerance_interval_py.svg?branch=master)](https://travis-ci.com/cjekel/tolerance_interval_py) [![codecov](https://codecov.io/gh/cjekel/tolerance_interval_py/branch/master/graph/badge.svg?token=K7JGW0PXHU)](https://codecov.io/gh/cjekel/tolerance_interval_py)

# Methods

Checkout the [documentation](https://jekel.me/tolerance_interval_py/index.html). This is what has been implemented so far:

## twoside

- normal
- normal_factor
- lognormal

## oneside

- normal
- lognormal
- non_parametric
- hanson_koopmans
- hanson_koopmans_cmh

# Requirements

```Python
"numpy >= 1.14.0"
"scipy >= 0.19.0"
"sympy >= 1.4"
"setuptools >= 38.6.0"
```
# Installation

```
python -m pip install toleranceinterval
```

or clone  and install from source

```
git clone https://github.com/cjekel/tolerance_interval_py
python -m pip install ./tolerance_interval_py
```

# Examples

The syntax follows ```(x, p, g)```, where ```x``` is the random sample, ```p``` is the percentile, and ```g``` is the confidence level. Here ```x``` can be a single set of random samples, or sets of random samples of the same size.

Estimate the 10th percentile to 95% confidence, of a random sample ```x``` using the Hanson and Koopmans 1964 method.

```python
import numpy as np
import toleranceinterval as ti
x = np.random.random(100)
bound = ti.oneside.hanson_koopmans(x, 0.1, 0.95)
print(bound)
```

Estimate the central 90th percentile to 95% confidence, of a random sample ```x``` assuming ```x``` follows a Normal distribution.

```python
import numpy as np
import toleranceinterval as ti
x = np.random.random(100)
bound = ti.twoside.normal(x, 0.9, 0.95)
print('Lower bound:', bound[:, 0])
print('Upper bound:', bound[:, 1])
```

All methods will allow you to specify sets of samples as 2-D numpy arrays. The caveat here is that each set must be the same size. This example estimates the 95th percentile to 90% confidence using the non-parametric method. Here ```x``` will be 7 random sample sets, where each set is of 500 random samples.

```python
import numpy as np
import toleranceinterval as ti
x = np.random.random((7, 500))
bound = ti.oneside.non_parametric(x, 0.95, 0.9)
# here bound will print for each set of n=500 samples 
print('Bounds:', bound)
```

# Changelog

Changes will be stored in [CHANGELOG.md](https://github.com/cjekel/tolerance_interval_py/blob/master/CHANGELOG.md).

# Contributing

All contributions are welcome! Please let me know if you have any questions, or run into any issues.

# License

MIT License

