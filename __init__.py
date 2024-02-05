"""
Modules:

- constants: provides various global constants.
- errors: provides custom error classes.
- typechecking: methods for type checking returning booleans.
- typevalidation: methods for type validation returning inputs or raising errors.
- typevariables: central collection of used variables for type hinting.
- utils: collection of reusable utility functions.
- warnings_: provides custom warning classes.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

from .constants import *
from .errors import *
from .typechecking import *
from .typevalidation import *
from .typevariables import *
from .utils import *
from .warnings_ import *

__all__ = (constants.__all__ +
           errors.__all__ +
           typechecking.__all__ +
           typevalidation.__all__ +
           typevariables.__all__ +
           utils.__all__ +
           warnings_.__all__)
