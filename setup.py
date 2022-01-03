from setuptools import setup, find_packages
packages = find_packages()
setup(
    name='toleranceinterval',
    version=open('toleranceinterval/VERSION').read().strip(),
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=packages,
    package_data={'toleranceinterval': ['VERSION']},
    py_modules=['toleranceinterval.__init__'],
    url='https://github.com/cjekel/tolerance_interval_py',
    license='MIT License',
    description='A small Python library for one-sided tolerance bounds and two-sided tolerance intervals.',  # noqa E501
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 0.19.0",
        "sympy >= 1.4",
        "setuptools >= 38.6.0",
    ],
    python_requires=">3.5",
)