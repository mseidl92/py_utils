"""
Classes:

- SecurityError: security errors (base class).
- ConnectionSecurityError: connection security errors.

- ConfigurationError: configuration errors (base class).
- ConfigurationFileError: configuration file errors.

- SimulationError: simulation errors (base class).

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

__all__ = ['SecurityError',
           'ConnectionSecurityError',
           'ConfigurationError',
           'ConfigurationFileError',
           'SimulationError']


class SecurityError(Exception):
    """
    Base class for security errors.
    """
    pass


class ConnectionSecurityError(SecurityError):
    """
    Class for connection security errors.
    """
    pass


class ConfigurationError(Exception):
    """
    Base class for configuration errors.
    """
    pass


class ConfigurationFileError(ConfigurationError):
    """
    Class for configuration file errors.
    """
    pass


class SimulationError(Exception):
    """
    Base class for the simulation errors.
    """
    pass
