"""
Methods:

- number: validates an input to be a number.
- non_negative_number: validates an input to be a non-negative number.
- positive_number: validates an input to be a positive number.
- non_positive_number: validates an input to be a non-positive number.
- negative_number: validates an input to be a negative number.
- numpy_array: validates an input to be a numpy array of given shape and type within given limits.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import numpy as np
from typing import Any
from types import NoneType

# project imports
from .typevars import TNum
from .typechecks import is_numeric, is_integer


def number(input_: Any,
           quantity_name: str = 'value',
           ) -> TNum:
    """
    Validates an input to be a (non-complex) number.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input of numeric type.
    :raises TypeError: if the input is not numeric.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    if not is_numeric(input_):
        raise TypeError('{} is not numeric'.format(quantity_name))
    return input_


def non_negative_number(input_: Any,
                        quantity_name: str = 'value',
                        ) -> TNum:
    """
    Validates an input to be a non-negative number.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-negative number.
    :raises ValueError: if the number is negative.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    number_ = number(input_, quantity_name)
    if number_ < 0.:
        raise ValueError('negative {}'.format(quantity_name))
    return number_


def positive_number(input_: Any,
                    quantity_name: str = 'value',
                    ) -> TNum:
    """
    Validates an input to be a positive number.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a positive number.
    :raises ValueError: if the number is non-positive.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    number_ = number(input_, quantity_name)
    if number_ <= 0.:
        raise ValueError('non-positive {}'.format(quantity_name))
    return number_


def non_positive_number(input_: Any,
                        quantity_name: str = 'value',
                        ) -> TNum:
    """
    Validates an input to be a non-positive number.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-positive number.
    :raises ValueError: if the number is positive.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    number_ = number(input_, quantity_name)
    if number_ > 0.:
        raise ValueError('positive {}'.format(quantity_name))
    return number_


def negative_number(input_: Any,
                    quantity_name: str = 'value',
                    ) -> TNum:
    """
    Validates an input to be a negative number.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a negative number.
    :raises ValueError: if the number is non-negative.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    number_ = number(input_, quantity_name)
    if number_ >= 0.:
        raise ValueError('non-negative {}'.format(quantity_name))
    return number_


def integer(input_: Any,
            quantity_name: str = 'value'
            ) -> int:
    """
    Validates an input to be an integer.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input of int type.
    :raises TypeError: if the input is not an integer.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    if not is_integer(input_):
        raise TypeError('{} is not an integer'.format(quantity_name))
    return input_


def non_negative_integer(input_: Any,
                         quantity_name: str = 'value',
                         ) -> TNum:
    """
    Validates an input to be a non-negative integer.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-negative integer.
    :raises ValueError: if the integer is negative.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    integer_ = integer(input_, quantity_name)
    if integer_ < 0.:
        raise ValueError('negative {}'.format(quantity_name))
    return integer_


def positive_integer(input_: Any,
                     quantity_name: str = 'value',
                     ) -> TNum:
    """
    Validates an input to be a positive integer.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a positive integer.
    :raises ValueError: if the integer is non-positive.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    integer_ = integer(input_, quantity_name)
    if integer_ <= 0.:
        raise ValueError('non-positive {}'.format(quantity_name))
    return integer_


def non_positive_integer(input_: Any,
                         quantity_name: str = 'value',
                         ) -> TNum:
    """
    Validates an input to be a non-positive integer.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-positive integer.
    :raises ValueError: if the integer is positive.
    """
    assert isinstance(quantity_name, str), 'quantity_name is not a string'

    integer_ = integer(input_, quantity_name)
    if integer_ > 0.:
        raise ValueError('positive {}'.format(quantity_name))
    return integer_


def negative_integer(input_: Any,
                     quantity_name: str = 'value',
                     ) -> TNum:
    """
    Validates an input to be a negative integer.

    :param input_: input to be validated.
    :param quantity_name: string of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a negative integer.
    :raises ValueError: if the integer is non-negative.
    """
    assert isinstance(quantity_name, str), 'quantity_name must be a string'

    integer_ = integer(input_, quantity_name)
    if integer_ >= 0.:
        raise ValueError('non-negative {}'.format(quantity_name))
    return integer_


def numpy_array(input_: Any,
                shape: None | tuple[None | int, ...] | list[tuple[None | int, ...]] = None,
                min_value: None | TNum = None,
                max_value: None | TNum = None,
                dtype: None | type = None,
                name_string: str = 'array',
                ) -> np.ndarray:
    """
    Validates an input to be a numpy array (or compatible) of a certain type.

    :param input_: input to be validated.
    :param shape: required shape as tuple or accepted shapes as list of tuples,
                  default is None to indicate no restrictions.
    :param min_value: smallest allowed value in the array, default is None to indicate no lower limit.
    :param max_value: largest allowed value in the array, default is None to indicate no upper limit.
    :param dtype: data type of array elements, only numeric types are allowed (promoting output to dtype),
                  default is None keeping the default type assigned by numpy for the input (still numeric only!).
    :param name_string: string of the arrays name, used in error message, default is generic term 'array'.
    :return: validated input, a numpy array of given shape and within given limits.
    :raises TypeError: if the input is not of proper shape, dtype is non-numeric or input cannot be cast to dtype.
    :raises ValueError: if the input array has (an) out of bounds value(s).
    """
    assert (isinstance(shape, NoneType) or isinstance(shape, tuple) and all([isinstance(dim, (NoneType, int))
                                                                             for dim in shape])
            or isinstance(shape, list)
            and all([isinstance(shape_, tuple)
                     and all([isinstance(dim, (NoneType, int)) for dim in shape_]) for shape_ in shape])), \
        'shape must be None, a tuple of int or a list of tuples of int'
    assert isinstance(min_value, NoneType) or is_numeric(min_value), 'min_value must be numeric or None'
    assert isinstance(max_value, NoneType) or is_numeric(max_value), 'max_value must be numeric or None'
    assert isinstance(dtype, NoneType) or isinstance(dtype, type) and np.can_cast(dtype, np.complex256), \
        'dtype must be None or a numeric type'
    assert isinstance(name_string, str), 'name_string must be a string'

    array_ = np.array(input_)  # to capture compatible inputs
    if (isinstance(shape, tuple) and not len(shape) == len(input_.shape)
            or not all([dim is None or dim == input_dim for dim, input_dim in zip(shape, input_.shape)])):
        raise TypeError('{} is not numpy array (compatible) of shape {}'.format(
            name_string, shape).replace('None', 'Any'))
    if (isinstance(shape, list)
            and not any([len(shape_) == len(input_.shape)
                         and all([dim is None or dim == input_dim
                                  for dim, input_dim in zip(shape_, input_.shape)]) for shape_ in shape])):
        raise TypeError('{} is not numpy array (compatible) of any of the shapes in {}'.format(
            name_string, shape).replace('None', 'Any'))
    if min_value is not None and np.min(array_) < min_value:
        raise ValueError('{} contains at least one value below minimum {}'.format(name_string, min_value))
    if max_value is not None and np.max(array_) > max_value:
        raise ValueError('{} contains at least one value above maximum {}'.format(name_string, max_value))
    if not np.can_cast(array_, dtype):
        raise TypeError('{} is not numpy array (compatible) of shape {}'.format(name_string, shape))
    return array_.astype(dtype)
