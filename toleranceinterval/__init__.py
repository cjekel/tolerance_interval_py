# from .toleranceinterval import *  # noqa F401
# import .hk as hk # noqa F401
# import .checks as checks  # noqa F401
# from .hk as hk
# import .checks as checks
# from .oneside import *  # noqa F401
# import .oneside as oneside
# from . import oneside
# from . import oneside
from . import oneside
from . import twoside
from . import hk
from . import checks
import os as _os  # noqa F401

# add rudimentary version tracking
__VERSION_FILE__ = _os.path.join(_os.path.dirname(__file__), 'VERSION')
__version__ = open(__VERSION_FILE__).read().strip()
