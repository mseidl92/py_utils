"""
Classes:

- FloatEquality: float equality tolerances.
- RGBColor: named RGB color values.
- HorizontalTextAlignment: enum of horizontal text alignment formats.
- VerticalTextAlignment: enum of vertical text alignment formats.
- WindowAnchor: enum of typical anchors of a window.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

__all__ = ['FloatEquality',
           'RGBColor',
           'HorizontalTextAlignment',
           'VerticalTextAlignment',
           'WindowAnchor']

# standard library imports
from enum import EnumMeta, Enum


class _ContainsEnumMeta(EnumMeta):
    """
    Inspired by
    https://stackoverflow.com/questions/43634618/how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch
    """
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class ContainsEnum(Enum, metaclass=_ContainsEnumMeta):
    """
    For typechecking purposes.
    """
    pass


class FloatEquality:
    """
    Tolerance values for float equality comparison
    """
    REL_TOL = 1e-5
    ABS_TOL = 1e-8


class RGBColor:
    """
    Named RGB color values.
    Basic RBG colors from `here <https://www.rapidtables.com/web/color/RGB_Color.html>`.

    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    LIME = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    SILVER = (192, 192, 192)
    GRAY = (128, 128, 128)
    MAROON = (128, 0, 0)
    OLIVE = (128, 128, 0)
    GREEN = (0, 128, 0)
    PURPLE = (128, 0, 128)
    TEAL = (0, 128, 128)
    NAVY = (0, 0, 128)


class HorizontalTextAlignment(ContainsEnum):
    """
    Standard horizontal text alignment modes.
    """
    LEFT = 0
    CENTERED = 1
    RIGHT = 2
    JUSTIFIED = 3


class VerticalTextAlignment(ContainsEnum):
    """
    Standard vertical text alignment modes.
    """
    TOP = 0
    CENTERED = 1
    BOTTOM = 2
    SPREAD = 3


class WindowAnchor(ContainsEnum):
    """
    Standard anchor points for a rectangular window.

    x--------x--------x
    |                 |
    x        x        x
    |                 |
    x--------x--------x

    """
    TOP_LEFT = 0
    LEFT_TOP = 0

    CENTER_LEFT = 1
    LEFT_CENTER = 1

    BOTTOM_LEFT = 2
    LEFT_BOTTOM = 2

    TOP_CENTER = 3
    CENTER_TOP = 3

    CENTER = 4
    CENTER_CENTER = 4

    BOTTOM_CENTER = 5
    CENTER_BOTTOM = 5

    TOP_RIGHT = 6
    RIGHT_TOP = 6

    CENTER_RIGHT = 7
    RIGHT_CENTER = 7

    BOTTOM_RIGHT = 8
    RIGHT_BOTTOM = 8
