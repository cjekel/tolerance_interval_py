from . import oneside  # noqa F401
from . import twoside  # noqa F401
from . import hk  # noqa F401
from . import checks  # noqa F401
import os as _os  # noqa F401

# add rudimentary version tracking
__VERSION_FILE__ = _os.path.join(_os.path.dirname(__file__), 'VERSION')
__version__ = open(__VERSION_FILE__).read().strip()
