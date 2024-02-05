"""
Classes:

- PhysicsWarning: physics warning (base class).
- EngineeringWarning: engineering warning (base class).
- SimulationWarning: simulation warning (base class).
- VisualizationWarning: complications during visualization warning (base class).

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

__all__ = ['PhysicsWarning',
           'EngineeringWarning',
           'SimulationWarning',
           'VisualizationWarning']


class PhysicsWarning(UserWarning):
    """
    Physics warnings indicating potential unphysical behaviors at runtime (base class).
    """
    pass


class EngineeringWarning(UserWarning):
    """
    Engineering warnings indicating potential issues with engineering constraints at runtime (base class)
    """
    pass


class SimulationWarning(UserWarning):
    """
    Simulation warnings indicating potential issues during simulation runs due to its setup or manipulation of internal
    states (base class).
    """
    pass


class VisualizationWarning(UserWarning):
    """
    Visualization warnings indicating issues during visualization (base class).
    """
    pass
