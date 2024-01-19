"""
Named type variables.
"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
from typing import TypeVar, Any, Optional
import numpy as np

# standard variable function arguments
TArgs = Optional[Any]
TKwargs = Optional[Any]

# composed data types
TNum = TypeVar('TNum', int, float, np.number)
T2DPoint = tuple[TNum, TNum]
T2Vector = tuple[TNum, TNum]
T3DPoint = tuple[TNum, TNum, TNum]
T3Vector = tuple[TNum, TNum, TNum]

# types of class instances (bound with string to prevent cyclic import issues)
TController = TypeVar('TController', bound='Controller')
TEnvironment = TypeVar('TEnvironment', bound='Environment')
TAgent = TypeVar('TAgent', bound='Agent')
