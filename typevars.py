"""
Named type variables.
"""

__author__ = 'Marius Seidl'
__date__ = '2023-08-14'
__version__ = '1.0'
__license__ = 'GPL'

from typing import TypeVar, Any, Optional
import numpy as np

# standard variable function arguments
TArgs = Optional[Any]
TKwargs = Optional[Any]

# composed data types
TNum = TypeVar('TNum', int, float, np.float64)
TPoint = tuple[TNum, TNum]
T2Vector = tuple[TNum, TNum]
TState = tuple[TNum, TNum, TNum, TNum, TNum, TNum]

# types of class instances
TController = TypeVar('TController', bound='Controller')
