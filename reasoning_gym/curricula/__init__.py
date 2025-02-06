from .algebra import *
from .algorithmic import *
from .arithmetic import *
from .code import *
from .cognition import *
from .games import *
from .geometry import *
from .graphs import *
from .logic import *

# Re-export all Curriculum classes
__all__ = []
for module in [
    algebra, algorithmic, arithmetic, code,
    cognition, games, geometry, graphs, logic
]:
    __all__.extend([name for name in module.__all__ if name.endswith('Curriculum')]) 