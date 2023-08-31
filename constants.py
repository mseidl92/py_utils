"""
Classes:

- FloatEquality: float equality tolerances.
- RGBColor: named RGB color values.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-08-14'
__version__ = '1.0'
__license__ = 'GPL'


class FloatEquality:
    """
    Tolerance values for float equality comparison
    """
    REL_TOL = 1e-5
    ABS_TOL = 1e-8


class RGBColor:
    """
    Named RGB color values.
    Basic RBG colors from `here <https://www.rapidtables.com/web/color/RGB_Color.html>`_.

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

