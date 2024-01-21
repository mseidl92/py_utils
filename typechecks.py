"""
Classes:

- ClassName: class description.

Methods: 

- method_name: method description.

"""

__author__ = 'Marius Seidl'
__date__ = '2024-01-09'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
from typing import Any
from types import NoneType
import numpy as np

# local library imports

# project imports
from .typevars import TNum


def is_numeric(input_: Any
               ) -> bool:
    """
    Checks if input value is a numeric (non-complex) type including numpy support.
    :param input_: value to be checked.
    :return: bool indicating if input is numeric.
    """
    return isinstance(input_, (int, float, np.number))


def is_integer(input_: Any
               ) -> bool:
    """
    Checks if input value is an integer including numpy support (signed and unsigned for numpy are true).
    :param input_: value to be checked.
    :return: bool indicating if input is an integer.
    """
    return isinstance(input_, (int, np.integer))


def is_numpy_array(input_: Any,
                   shape: None | tuple[None | int, ...] | list[tuple[None | int, ...]] = None,
                   min_value: None | TNum = None,
                   max_value: None | TNum = None,
                   dtype: None | type | list[type] = None,
                   ) -> bool:
    """
    Checks if input is a numpy array of a certain conditions.

    :param input_: input to be checked
    :param shape: required shape as tuple of integer or accepted shapes as list of tuples of integers,
                  indicate required dimensions without restricted size as None
                  (e.g. any 2D array is (None, None), a 2 column array with any number of rows is (None, 2))
                  default is None to indicate no restrictions.
    :param min_value: smallest allowed value in the array, default is None to indicate no lower limit.
    :param max_value: largest allowed value in the array, default is None to indicate no upper limit.
    :param dtype: data type of array elements (only numeric types are allowed), default is None indicating no type check
    :return: bool indicating if input is a numpy array of given conditions.
    """
    assert (isinstance(shape, NoneType) or is_numpy_array_shape(shape, allow_none=True)
            or isinstance(shape, list) and all([is_numpy_array_shape(shape_, allow_none=True) for shape_ in shape])), \
        'shape must be None, a tuple of int or a list of tuples of int'
    assert isinstance(min_value, NoneType) or is_numeric(min_value), 'min_value must be numeric or None'
    assert isinstance(max_value, NoneType) or is_numeric(max_value), 'max_value must be numeric or None'
    assert (isinstance(dtype, NoneType) or is_numpy_dtype(dtype)
            or isinstance(dtype, list) and all([is_numpy_dtype(dtype_) for dtype_ in dtype])), \
        'dtype must be None, a numeric type or a list of numeric types'

    if not isinstance(input_, np.ndarray):
        return False
    if (isinstance(shape, tuple) and not len(shape) == len(input_.shape)
            or not all([dim is None or dim == input_dim for dim, input_dim in zip(shape, input_.shape)])):
        return False
    if (isinstance(shape, list)
            and not any([len(shape_) == len(input_.shape)
                         and all([dim is None or dim == input_dim
                                  for dim, input_dim in zip(shape_, input_.shape)]) for shape_ in shape])):
        return False
    if min_value is not None and np.min(input_) < min_value:
        return False
    if max_value is not None and np.max(input_) > max_value:
        return False
    if isinstance(dtype, type) and not input_.dtype.type == dtype:
        return False
    if isinstance(dtype, list) and not any([input_.dtype == dtype_ for dtype_ in dtype]):
        return False
    return True


def is_numpy_array_shape(input_: Any,
                         allow_none: bool = False
                         ) -> bool:
    """
    Checks if the input is a valid shape for a numpy array.

    :param input_: input to be checked.
    :param allow_none: allows None as members of shape tuple, indicating any size in that dimension.
    :return: boolean indicating if the input is a valid shape.
    """

    return (isinstance(input_, tuple)
            and all([isinstance(dimension, (NoneType, int) if allow_none else int) for dimension in input_]))


def is_numpy_dtype(input_: Any
                   ) -> bool:
    """
    Checks if the input is a valid dtype (or compatible) for a numpy array.

    :param input_: input to be checked.
    :return: boolean indicating if the input is a valid dtype.
    """
    return isinstance(input_, type) and np.can_cast(input_, np.complex256)


def is_cv_image(input_: Any
                ) -> bool:
    """
    Checks if the input is a numpy array containing a valid openCV image.
    Monochrome, BGR and BGRA images are considered valid.
    Images that are rescaled (float type) during displaying are considered invalid.

    :param input_: input to be checked.
    :return: boolean indicating if input is a numpy array representing a valid openCV image.
    """
    return (is_numpy_array(input_, shape=[(None, None), (None, None, 3), (None, None, 4)],
                           dtype=[np.uint8, np.int8, np.uint16, np.int16])
            or is_numpy_array(input_, shape=[(None, None), (None, None, 3), (None, None, 4)],
                              min_value=0., max_value=1., dtype=[np.float32, np.float64]))
