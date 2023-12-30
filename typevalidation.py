"""
Methods:

- number: validates an input to be a number.
- non_negative_number: validates an input to be a non-negative number.
- positive_number: validates an input to be a positive number.
- non_positive_number: validates an input to be a non-positive number.
- negative_number: validates an input to be a negative number.
- numpy_array: validates an input to be a numpy array of given shape within given limits.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import numpy as np
from typing import Any

# project imports
from .typevars import TNum
from .utils import is_numeric


def number(input_: Any
           ) -> TNum:
    """
    Validates an input to be a number.

    :param input_: input to be validated.
    :return: validated input of numeric type.
    :raises TypeError: if the input is not numeric.
    """
    if not is_numeric(number):
        raise TypeError('input is not numeric')
    return input_


def non_negative_number(input_: Any,
                        quantity_name: str = 'value',
                        ) -> TNum:
    """
    Validates an input to be a non-negative number.

    :param input_: input to be validated.
    :param quantity_name: sting of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-negative number.
    :raises ValueError: if the number is negative.
    """
    number_ = number(input_)
    if number_ < 0.:
        raise ValueError('negative {}'.format(quantity_name))
    return number_


def positive_number(input_: Any,
                    quantity_name: str = 'value',
                    ) -> TNum:
    """
    Validates an input to be a positive number.

    :param input_: input to be validated.
    :param quantity_name: sting of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a positive number.
    :raises ValueError: if the number is non-positive.
    """
    number_ = number(input_)
    if number_ <= 0.:
        raise ValueError('non-positive {}'.format(quantity_name))
    return number_


def non_positive_number(input_: Any,
                        quantity_name: str = 'value',
                        ) -> TNum:
    """
    Validates an input to be a non-positive number.

    :param input_: input to be validated.
    :param quantity_name: sting of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a non-positive number.
    :raises ValueError: if the number is positive.
    """
    number_ = number(input_)
    if number_ > 0.:
        raise ValueError('positive {}'.format(quantity_name))
    return number_


def negative_number(input_: Any,
                    quantity_name: str = 'value',
                    ) -> TNum:
    """
    Validates an input to be a negative number.

    :param input_: input to be validated.
    :param quantity_name: sting of the represented quantities name, used in error message,
                          default is generic term 'value'.
    :return: validated input, a negative number.
    :raises ValueError: if the number is non-negative.
    """
    number_ = number(input_)
    if number_ >= 0.:
        raise ValueError('non-negative {}'.format(quantity_name))
    return number_


def numpy_array(input_: Any,
                shape: None | tuple | list[tuple, ...] = None,
                min_value: float = np.nan,
                max_value: float = np.nan,
                name_string: str = 'array',
                ) -> np.ndarray:
    """
    Validates an input to be a numpy array (or compatible) of a certain type.

    :param input_: input to be validated.
    :param shape: required shape as tuple or accepted shapes as list of tuples,
                  default is None to indicate no restrictions.
    :param min_value: smallest allowed value in the array, default is np.nan to indicate no lower limit.
    :param max_value: largest allowed value in the array, default is np.nan to indicate no upper limit.
    :param name_string: string of the arrays name, used in error message, default is generic term 'array'.
    :return: validated input, a numpy array of given shape and within given limits.
    :raises TypeError: if the input is not of the proper shape.
    :raises ValueError: if the input array has out of bounds value(s).
    """
    array_ = np.array(input_)  # to capture compatible inputs
    type_error_str = '{} not numpy array (or compatible) of shape(s) {}'.format(name_string, shape)
    if isinstance(shape, tuple) and not array_.shape == shape:
        raise TypeError(type_error_str)
    if isinstance(shape, list) and not any([array_.shape == s for s in shape]):
        raise TypeError(type_error_str)
    if not np.isnan(min_value) and np.min(array_) < min_value:
        raise ValueError('{} contains at least one value below minimum {}'.format(name_string, min_value))
    if not np.isnan(max_value) and np.max(array_) > max_value:
        raise ValueError('{} contains at least one value above maximum {}'.format(name_string, max_value))
    return array_
