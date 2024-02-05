"""
Classes:

- ContainsEnum: Enum type that supports pythons "in" operation.


"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

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
    Enum type that supports pythons "in" operation.
    """
    pass
