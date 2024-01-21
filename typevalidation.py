"""
Methods:

- or_: chaining validation functions by or.
- and_: chaining validation functions by and. USE WITH CAUTION: useful debugging statements not guaranteed.
- not_: negating validation functions. USE WITH CAUTION: useful debugging statements not guaranteed.
- number: validate input to be (non-complex) numeric.
- integer: validate input to be an integer.
- less: validate numeric input to be less than an upper limit.
- less_equal: validate numeric input to be less than or equal to an upper limit.
- equal: validate numeric input to be equal to a comparison value.
- not_equal: validate numeric input to be not equal to a comparison value.
- greater: validate numeric input to be greater than a lower limit.
- greater_equal: validate numeric input to be greater than or equal to a lower limit.
- numpy_array: validate a numpy array or compatible with restricted shape, limits and data type.

"""

__author__ = 'Marius Seidl'
__date__ = '2023-12-21'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import numpy as np
from typing import Any, TypeVar, Callable, Protocol
from types import NoneType
import inspect
import re
import traceback

# project imports
from .typevars import TNum
from .typechecks import is_numeric, is_integer, is_numpy_array_shape, is_numpy_dtype


# TODO, that works in mypy but not with the pycharm typechecker, wait for update to use it instead of Callable[..., T]
class _ValidationFunction(Protocol):
    """
    Protocol defining the callback type for a validation function.
    """
    def __call__(self, input_: Any, /, *, variable_name: str | None = None) -> Any: ...


def _is_validation_function(input_: Any
                            ) -> bool:
    """
    Helper function to check if an input is a valid validation function.

    Validation functions have one positional-only parameter without default value (the input to be validated)
    and at least one keyword-only parameter 'variable_name' with default value None and type string.
    It is used to print more meaningful debugging information.
    All other parameters must be bound or must have a default value, to which they will be evaluated during validation.
    """
    if not callable(input_):
        return False
    parameters = inspect.signature(input_).parameters
    if not len(parameters) > 2:
        return False
    # input is first parameter
    for parameter in parameters.values():
        # mapping order is guaranteed but indexing is not possible -> iterate and break for first argument
        if not parameter.default == parameter.empty and not parameter.kind == parameter.POSITIONAL_ONLY:
            return False
        break
    # input is only parameter without default value
    if not len([parameter for parameter in parameters.values()
                if not parameter.default == parameter.empty]) == 1:
        return False
    # check if keyword parameter variable_name exists and has correct properties
    try:
        if not parameters['variable_name'].kind == parameters['variable_name'].KEYWORD_ONLY:
            return False
        if not parameters['variable_name'].default is None:
            return False
        if not parameters['variable_name'].annotation == str | None:
            return False
    except KeyError:
        return False
    return True


def _get_parameter_name(parameter_position: int = 1
                        ) -> str:
    """
    Get the name of the variable assigned to a parameter of the function calling this as a string.
    Uses stack inspection and should be used for debugging messages only!
    :param parameter_position: The position of the parameter in the function signature (starting at 1).
    :return: The name of the variable assigned to the parameter_positions parameter of the function calling this as
             a string or the value if the parameter was given directly or the empty string if the calling function has
             not that many parameters.
    """
    assert isinstance(parameter_position, int) and parameter_position > 0, \
        'parameter_position must be a positive integer'

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-3]
    try:
        return re.compile(r'\((.*?)\).*$').search(code).groups()[parameter_position - 1]
    except IndexError:
        return ''


# TODO replace with local generic typing once updated to py 3.12 (e.g. def or_[T](...))
T = TypeVar('T')


def or_(input_: Any,
        *validation_functions: Callable[..., T],
        ) -> T:
    """
    Validates an input against provided validation functions. Successful validation occurs if any function validates
    the input (or-functionality). Validation does not short-circuit.

    :param input_: input to be validated.
    :param validation_functions: any number of callable validation functions. Functions must be callable with only
                                 one parameter. Consider binding others with a lambda-expression upon calling.
    :return: validated input.
    :raises Exception: if all validation functions fail to validate the input.
    """
    assert all([_is_validation_function(validation_function) for validation_function in validation_functions]), \
        ('validation_functions must be callables with only one parameter that does not have a default value. '
         'Consider binding with a lambda-expression.')

    exceptions: list[Exception] = []
    for validation_function in validation_functions:
        try:
            return validation_function(input_)
        except Exception as e:
            exceptions.append(e)
    # TODO use ExceptionGroup once updated to py 3.11
    raise [Exception('or-validation failed. All exceptions raised.')] + exceptions


def and_(input_: Any,
         *validation_functions: Callable[..., T],
         ) -> T:
    """
    Validates an input against provided validation functions. Successful validation occurs if all functions validate
    the input (and-functionality). Validation does not short-circuit.

    ATTENTION: Consider using nested function calls (e.g. less(integer(input_, 0)) instead of using this function,
    if at all possible to prevent nested stack inspection happening. That will give more meaningful error messages!


    :param input_: input to be validated.
    :param validation_functions: any number of callable validation functions. Functions must be callable with only
                                 one parameter. Consider binding others with a lambda-expression upon calling.
    :return: validated input.
    :raises Exception: if any validation function fails to validate the input.
    """
    assert all([_is_validation_function(validation_function) for validation_function in validation_functions]), \
        ('validation_functions must be callables with only one parameter that does not have a default value. '
         'Consider binding with a lambda-expression.')

    exceptions: list[Exception] = []
    for validation_function in validation_functions:
        try:
            validation_function(input_)
        except Exception as e:
            exceptions.append(e)
    if exceptions:
        # TODO use ExceptionGroup once updated to py 3.11
        raise [Exception('and-validation failed. Exception(s) raised.')] + exceptions
    return input_


def not_(input_: Any,
         validation_function: Callable[..., T]  # _ValidationFunction,
         ) -> T:
    """

    """
    assert _is_validation_function(validation_function), \
        ('validation_function must be callables with only one parameter that does not have a default value. '
         'Consider binding with a lambda-expression.')
    try:
        validation_function(input_, _get_parameter_name(1))
    except:
        return input_
    raise Exception('not-validation failed. No exceptions raised by {}'.format(_get_parameter_name(2)))


def number(input_: Any,
           /,
           *,
           variable_name: str | None = None
           ) -> TNum:
    """
    Validates an input to be a (non-complex) number.

    :param input_: input to be validated.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input of numeric type.
    :raises TypeError: if the input is not numeric.
    """
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not is_numeric(input_):
        raise TypeError('{} is not numeric.'.format(_get_parameter_name(1) if variable_name is None else variable_name))
    return input_


def integer(input_: Any,
            /,
            *,
            variable_name: str | None = None
            ) -> int:
    """
    Validates an input to be an integer.

    :param input_: input to be validated.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input of int type.
    :raises TypeError: if the input is not an integer.
    """
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not is_integer(input_):
        raise TypeError('non-integer {}'.format(_get_parameter_name(1) if variable_name is None else variable_name))
    return input_


def less(number_: TNum,
         /,
         upper_limit: TNum,
         *,
         variable_name: str | None = None
         ) -> TNum:
    """
    Validates a numeric input to be less than an upper limit.

    :param number_: numeric input to be validated.
    :param upper_limit: upper limit below which the input is valid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a number less than upper_limit.
    :raises ValueError: if the number is greater than or equal to upper_limit;
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(upper_limit), 'upper_limit must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not number_ < upper_limit:
        raise ValueError('{} is not less than {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, upper_limit))
    return number_


def less_equal(number_: TNum,
               /,
               upper_limit: TNum,
               *,
               variable_name: str | None = None
               ) -> TNum:
    """
    Validates a numeric input to be less than or equal to an upper limit.

    :param number_: numeric input to be validated.
    :param upper_limit: upper limit below and at which the input is valid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a number less than or equal to upper_limit.
    :raises ValueError: if the number is greater than upper_limit.
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(upper_limit), 'upper_limit must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not number_ <= upper_limit:
        raise ValueError('{} is not less than or equal to {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, upper_limit))
    return number_


def equal(number_: TNum,
          /,
          comparison_value: TNum,
          *,
          variable_name: str | None = None
          ) -> TNum:
    """
    Validates a numeric input to be equal to a comparison value.

    :param number_: numeric input to be validated.
    :param comparison_value: value at which the input is valid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a value equal to comparison_value.
    :raises ValueError: if the value is not equal to comparison_value.
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(comparison_value), 'comparison_value must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not number_ == comparison_value:
        raise ValueError('{} is not equal to {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, comparison_value))
    return number_


def not_equal(number_: TNum,
              /,
              comparison_value: TNum,
              *,
              variable_name: str | None = None
              ) -> TNum:
    """
    Validates a numeric input to be not equal to a comparison value.

    :param number_: numeric input to be validated.
    :param comparison_value: value at which the input is invalid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a value not equal to comparison_value.
    :raises ValueError: if the value is equal to comparison_value.
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(comparison_value), 'comparison_value must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if number_ == comparison_value:
        raise ValueError('{} is equal to {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, comparison_value))
    return number_


def greater(number_: TNum,
            /,
            lower_limit: TNum,
            *,
            variable_name: str | None = None
            ) -> TNum:
    """
    Validates a numeric input to be greater than a lower limit.

    :param number_: numeric input to be validated.
    :param lower_limit: lower limit above which the input is valid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a number greater than lower_limit.
    :raises ValueError: if the number is less than or equal to lower_limit;
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(lower_limit), 'lower_limit must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not number_ > lower_limit:
        raise ValueError('{} is not greater than {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, lower_limit))
    return number_


def greater_equal(number_: TNum,
                  /,
                  lower_limit: TNum,
                  *,
                  variable_name: str | None = None
                  ) -> TNum:
    """
    Validates a numeric input to be greater than or equal to a lower limit.

    :param number_: numeric input to be validated.
    :param lower_limit: lower limit below and at which the input is valid.
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a number greater than or equal to lower_limit.
    :raises ValueError: if the number is less than lower_limit.
    """
    assert is_numeric(number_), 'number_ must be numeric'
    assert is_numeric(lower_limit), 'lower_limit must be numeric'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    if not number_ >= lower_limit:
        raise ValueError('{} is not greater than or equal to {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, lower_limit))
    return number_


def numpy_array(input_: Any,
                /,
                shape: None | tuple[None | int, ...] | list[tuple[None | int, ...]] = None,
                min_value: None | TNum = None,
                max_value: None | TNum = None,
                dtype: None | type = None,
                *,
                variable_name: str | None = None
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
    :param variable_name: string of the variable to be validated, used in error message. Default is None triggering
                          name detection by stack inspection. Use internally when stack inspection will fail due to
                          different depth of call in stack.
    :return: validated input, a numpy array of given shape and within given limits.
    :raises TypeError: if the input is not of proper shape, dtype is non-numeric or input cannot be cast to dtype.
    :raises ValueError: if the input array has (an) out of bounds value(s).
    """
    assert (isinstance(shape, NoneType) or is_numpy_array_shape(shape, allow_none=True)
            or isinstance(shape, list) and all([is_numpy_array_shape(shape_, allow_none=True) for shape_ in shape])), \
        'shape must be None, a tuple of int or a list of tuples of int'
    assert isinstance(min_value, NoneType) or is_numeric(min_value), 'min_value must be numeric or None'
    assert isinstance(max_value, NoneType) or is_numeric(max_value), 'max_value must be numeric or None'
    assert (isinstance(dtype, NoneType) or is_numpy_dtype(dtype)
            or isinstance(dtype, list) and all([is_numpy_dtype(dtype_) for dtype_ in dtype])), \
        'dtype must be None, a numeric type or a list of numeric types'
    assert isinstance(variable_name, (str, NoneType)), 'variable_name must be a string or None'

    array_ = np.array(input_)  # to capture compatible inputs
    if (isinstance(shape, tuple) and not len(shape) == len(input_.shape)
            or not all([dim is None or dim == input_dim for dim, input_dim in zip(shape, input_.shape)])):
        raise TypeError('{} is not numpy array (compatible) of shape {}'.format(
            _get_parameter_name(1) if variable_name is None else variable_name, shape).replace('None', 'Any'))
    if (isinstance(shape, list)
            and not any([len(shape_) == len(input_.shape)
                         and all([dim is None or dim == input_dim
                                  for dim, input_dim in zip(shape_, input_.shape)]) for shape_ in shape])):
        raise TypeError('{} is not numpy array (compatible) of any of the shapes in {}'.format(
            _get_parameter_name(1) if variable_name is None else variable_name, shape).replace('None', 'Any'))
    if min_value is not None and np.min(array_) < min_value:
        raise ValueError('{} contains at least one value below minimum {}'.
                         format(_get_parameter_name(1) if variable_name is None else variable_name, min_value))
    if max_value is not None and np.max(array_) > max_value:
        raise ValueError('{} contains at least one value above maximum {}'
                         .format(_get_parameter_name(1) if variable_name is None else variable_name, max_value))
    if not np.can_cast(array_, dtype):
        raise TypeError('{} is not numpy array (compatible) castable to {}'
                        .format(_get_parameter_name(1) if variable_name is None else variable_name,
                                np.dtype(dtype).name))
    return array_.astype(dtype)



