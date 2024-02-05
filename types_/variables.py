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

# composed data types_
TNum = TypeVar('TNum', int, float, np.number)
T2DPoint = tuple[TNum, TNum]
T2Vector = tuple[TNum, TNum]
T3DPoint = tuple[TNum, TNum, TNum]
T3Vector = tuple[TNum, TNum, TNum]
TSequence = TypeVar('TSequence', list, tuple, range, np.ndarray)
