"""
Methods:

- Mathematical:

  - is_close: robust float equality.
  - clip: limit input to interval

- Geometry:

  - wrap_to_pi: map angle to [-pi, pi) interval
  - wrap_to_2pi: map angle to [0, 2*pi) interval
  - rotation_matrix_2d: creates rotation matrix in 2 dimensions.
  - polygon_outline_to_vertices: convert length and inner angles of polygon to vertex coordinates.
  - polygon_centroid: get centroid from polygon vertices list.

- Typechecking:

  - is_numeric: numeric type check.

- File Handling:

  - load_json: json file loading helper.


"""

__author__ = 'Marius Seidl'
__date__ = '2023-08-14'
__version__ = '1.0'
__license__ = 'GPL'

# standard library imports
import numpy as np
import math
import numbers
import json
from typing import Any

# project imports
from utils.constants import FloatEquality
from utils.typevars import TNum, TPoint


"""Mathematical"""


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


"""Geometry"""


def wrap_to_pi(angle: TNum
               ) -> TNum:
    """
    Maps an angle in radians onto the interval [-pi, pi).

    :param angle: the angle to be wrapped to by.
    :return: wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_to_2pi(angle: TNum
                ) -> TNum:
    """
    Maps an angle in radians onto the interval [0, 2*pi).

    :param angle: the angle to be wrapped to by.
    :return: wrapped angle in radians.
    """
    return angle % (2 * np.pi)


def rotation_matrix_2d(angle: float
                       ) -> np.ndarray:
    """
    Creates a 2D rotation matrix from an angle.

    :param angle: the angle in radians.
    :return: numpy array 2x2 rotation matrix-
    """
    s, c = np.sin(angle), np.cos(angle)
    return np.array([[c, -s], [s, c]])


def polygon_outline_to_vertices(length: list[TNum],
                                angles: list[TNum],
                                degrees=False
                                ) -> np.ndarray:
    """
    Converts a list of side length and angles
    to a list of (x,y) vertices of a polygon.

    :param length: list of side length of the polygon in clockwise order, starting at (0, 0).
    :param angles: list of inside angles of the polygon in clockwise order, starting with first at (0, 0).
    :param degrees: boolean flag indicating if angles are in degree. Default is radians (=False).
    :return: list of vertices in scale indicated by length.
    """

    # convert to numpy and radians
    length = np.array(length)
    angles = np.array(angles)
    if degrees:
        angles = np.deg2rad(angles)

    # check if angles sum error needs correction and apply deviation equally
    angles += (np.pi * (angles.shape[0] - 2) - np.sum(angles)) / angles.shape[0]

    # convert to vertices
    vertices = np.cumsum(np.array([np.cos(np.cumsum(angles - np.pi) + np.pi),
                                   np.sin(np.cumsum(angles - np.pi) + np.pi)]).T * length[:, np.newaxis], axis=0)

    # use missmatch between [0, 0] and calculated last vertex to estimate x- and y-stretch factor
    vertices *= 1 - (vertices[-1, :] / (np.max(vertices, axis=0) - np.min(vertices, axis=0)))
    vertices[-1, :] = [0, 0]

    return vertices


def polygon_centroid(vertices: list[TPoint] | np.ndarray
                     ) -> TPoint:
    """
    Calculates coordinates of the centroid of a polygon

    :param vertices: vertices of the polygon, either as a list of points or an (N,2) numpy array.
    :return: coordinates of the centroid.
    """
    vertices = np.array(vertices)
    vertices_shoelace = vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]
    shoelace_factor = 1 / (3 * np.sum(vertices_shoelace))
    centroid_x = shoelace_factor * np.sum((vertices[:-1, 0] + vertices[1:, 0]) * vertices_shoelace)
    centroid_y = shoelace_factor * np.sum((vertices[:-1, 1] + vertices[1:, 1]) * vertices_shoelace)
    return centroid_x, centroid_y


"""Typechecking"""


def is_numeric(input_: Any
               ) -> bool:
    """
    Checks if input value is a numeric type.
    :param input_: value to be checked.
    :return: bool indicating if input is numeric.
    """
    return isinstance(input_, numbers.Number)


"""File Handling"""


def load_json(file_path
              ) -> dict:
    """
    Loads json file.

    :param file_path: path to the json file.
    :return: dict containing json file content
    """
    with open(file_path) as fp:
        return json.load(fp)



