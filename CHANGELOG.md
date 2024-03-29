# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2023-03-26
### Changed
- Fixed a bug introduced in `1.0.2` where `checks.assert_2d_sort` was not sorting

## [1.0.2] - 2023-03-25
### Changed
- Fixed a bug where a sort function was modifying the input data array. This could have unintended consequence for a user without their knowledge. See [PR](https://github.com/cjekel/tolerance_interval_py/pull/7). Thanks to Jed Ludlow](https://github.com/jedludlow)

## [1.0.1] - 2022-02-24
### Added
- Fix docstring for oneside.non_parametric thanks to [Jed Ludlow](https://github.com/jedludlow) 

## [1.0.0] - 2022-01-03
### Added
- exact two-sided normal method thanks to [Jed Ludlow](https://github.com/jedludlow) 
- normal_factor method thanks to [Jed Ludlow](https://github.com/jedludlow)
### Changed
- Fixed references listed in documentation
### Removed
- Python 2.X is no longer supported. Python 3.6 is the minimum supported version. 

## [0.0.3] - 2020-05-22
### Changed
- Docstrings, documentation, and readme had the wrong percentile values for many examples. I've corrected these examples. Sorry for the confusion this may have caused.

## [0.0.2] - 2019-11-13
### Added
- setuptools is now listed in the requirements

## [0.0.1] - 2019-11-03
### Added
- Everything you've seen so far!
