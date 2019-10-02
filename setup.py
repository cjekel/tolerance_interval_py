from distutils.core import setup
setup(
    name='tolinterval',
    version=open('tolinterval/VERSION').read().strip(),
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['tolinterval'],
    package_data={'tolinterval': ['VERSION']},
    url='https://github.com/cjekel/tolerance_interval_py',
    license='MIT License',
    description='Tolerance intervals in Python',
    long_description=open('README.rst').read(),
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 0.19.0",
        "sympy >= 1.4",
    ]
)