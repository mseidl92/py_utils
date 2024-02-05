"""
Classes:

- FloatEquality: float equality tolerances.

Functions:

- is_close: robust float equality.
- clip: limit input to interval

"""

__author__ = 'Marius Seidl'
__date__ = '2024-02-05'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import math

# project imports
from ..types_.variables import TNum


class FloatEquality:
    """
    Tolerance values for float equality comparison
    """
    REL_TOL = 1e-5
    ABS_TOL = 1e-8


def is_close(x: float,
             y: float,
             ) -> bool:
    """
    Determines if two floats are close based on tolerances.
    Use for robust float equality comparison.

    :param x: first float to compare.
    :param y: second float to compare.
    :return: bool indicating if x and y are close
    """
    return math.isclose(x, y, rel_tol=FloatEquality.REL_TOL, abs_tol=FloatEquality.ABS_TOL)


def clip(x: TNum,
         min_value: TNum,
         max_value: TNum,
         ) -> TNum:
    """
    Returns value, but moved to closer limiting value if outside min-max range.

    :param x: value to be clipped.
    :param min_value: lower clipping bound.
    :param max_value: upper clipping bound.
    :return: clipped numeric value
    """
    return max(min_value, min(max_value, x))
