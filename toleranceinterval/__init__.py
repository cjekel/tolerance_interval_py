# from .toleranceinterval import *  # noqa F401
# import .hk  # noqa F401
# import .checks  # noqa F401
# import .hk as hk
# import .checks as checks
import os  # noqa F401

# add rudimentary version tracking
VERSION_FILE = os.path.join(os.path.dirname(__file__), 'VERSION')
__version__ = open(VERSION_FILE).read().strip()
