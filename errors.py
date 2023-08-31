"""
Classes:

- SecurityError: connection security errors.
- ConfigurationError: config file errors.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-08-14'
__version__ = '1.0'
__license__ = 'GPL'


class SecurityError(Exception):
    """
    Base class for connection security errors.
    """
    pass


class ConfigurationError(Exception):
    """
    Base class for configuration errors.
    """
    pass
