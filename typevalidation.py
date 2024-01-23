"""
Methods:

- bind: wrapper for partial of functools.
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
- sequence: validate input to be a sequence of certain properties.
- tuple_: validate input to be a tuple of certain properties.
- list_: validate a list to be a list with certain properties.
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
from functools import partial

# project imports
from typevariables import TNum, TSequence, TArgs, TKwargs
import typechecking as check


# TODO that works in mypy but not with the pycharm typechecker, wait for update to use it instead of Callable[..., U]
# TODO incorporate required default values?
class _ValidationFunction(Protocol):
    """
    Protocol defining the callback type for a validation function.
    """

    def __call__(self, input_: Any, /, *args, **kwargs) -> Any: ...


def _is_validation_function(input_: Any
                            ) -> bool:
    """
    Helper function to check if an input is a valid validation function.

    Validation functions have one positional-only parameter without default value (the input to be validated)
    All other parameters must be bound or must have a default value, to which they will be evaluated during validation.
    """
    if not callable(input_):
        return False
    parameters = inspect.signature(input_).parameters
    if len(parameters) < 1:
        return False
    # input is first parameter
    for parameter in parameters.values():
        # mapping order is guaranteed but indexing is not possible -> iterate and break for first argument
        if parameter.default != parameter.empty or parameter.kind != parameter.POSITIONAL_ONLY:
            return False
        break
    # input is only parameter without default value
    if len([parameter for parameter in parameters.values() if parameter.default == parameter.empty]) != 1:
        return False
    return True


def bind(function: Callable,
         /,
         *args: TArgs,
         **kwargs: TKwargs
         ) -> Callable:
    """
    Wrapper around functools.partial.
    Use to bind parameters to a functions, so it fits the signature of a validation function.

    Validation functions have one positional-only parameter without default value (the input to be validated)
    and at least one keyword-only parameter 'variable_name' with default value None and type str | None.
    It is used to print more meaningful debugging information.
    All other parameters must be bound or must have a default value, to which they will be evaluated during validation.
    """
    return partial(function, *args, **kwargs)


# TODO replace with local generic typing once updated to py 3.12 (e.g. def or_[T](...))
U = TypeVar('U')


def or_(input_: Any,
        /,
        *validation_functions: Callable[..., U]  # _ValidationFunction,
        ) -> U:
    """
    Validates an input against provided validation functions. Successful validation occurs if any function validates
    the input (or-functionality). Validation does short-circuit.

    :param input_: input to be validated.
    :param validation_functions: any number of callable validation functions.
                                 (see _is_validation_function for specifications)
    :return: validated input.
    :raises Exception: if all validation functions fail to validate the input.
    """
    assert all([_is_validation_function(validation_function) for validation_function in validation_functions]), \
        ('validation_functions must be callables with only one parameter without default value (positional-only). '
         'Consider binding other parameters with functools.partial or the wrapper called \'bind\' from this module.')

    exceptions: list[Exception] = []
    for validation_function in validation_functions:
        try:
            return validation_function(input_)
        except Exception as e:
            exceptions.append(e)
    # TODO use ExceptionGroup once updated to py 3.11
    raise Exception('or-validation failed. All validations raised exceptions:', exceptions)


def and_(input_: Any,
         /,
         *validation_functions: Callable[..., U]  # _ValidationFunction,
         ) -> U:
    """
    Validates an input against provided validation functions. Successful validation occurs if all functions validate
    the input (and-functionality). Validation does not short-circuit, to provide all exception raised.

    ATTENTION: Consider using nested function calls (e.g. less(integer(input_, 0)) instead of using this function,
    if at all possible to prevent nested stack inspection happening. That will give more meaningful error messages!


    :param input_: input to be validated.
    :param validation_functions: any number of callable validation functions.
                                 (see _is_validation_function for specifications)
    :return: validated input.
    :raises Exception: if any validation function fails to validate the input.
    """
    assert all([_is_validation_function(validation_function) for validation_function in validation_functions]), \
        ('validation_functions must be callables with only one parameter without default value (positional-only). '
         'Consider binding other parameters with functools.partial or the wrapper called \'bind\' from this module.')

    exceptions: list[Exception] = []
    for validation_function in validation_functions:
        try:
            validation_function(input_)
        except Exception as e:
            exceptions.append(e)
    if exceptions:
        # TODO use ExceptionGroup once updated to py 3.11
        raise Exception('and-validation failed. Exception(s) raised:', exceptions)
    return input_


def not_(input_: Any,
         /,
         validation_function: Callable[..., U]  # _ValidationFunction,
         ) -> U:
    """
    Validates an input against provided validation functions. Successful validation occurs if the validation
    function fails to validate the input (not-functionality).

    ATTENTION: Consider using a function that can positively validate the input instead of using this function,
    if at all possible. This function will not be able to give meaningful debugging messages.

    :param input_: input to be validated.
    :param validation_function: a callable validation function.
                                 (see _is_validation_function for specifications)

    """
    assert _is_validation_function(validation_function), \
        ('validation_functions must be callables with only one parameter without default value (positional-only). '
         'Consider binding other parameters with functools.partial or the wrapper called \'bind\' from this module.')

    try:
        validation_function(input_)
    except:
        return input_
    raise Exception('not-validation failed. No exceptions raised.')


def none_(input_: Any,
          /
          ) -> None:
    """
    Validates an input to be None.

    :param input_: input to be validated.
    :return: validated input being None.
    :raises TypeError: if the input is not None.
    """
    if input_ is not None:
        raise TypeError('value is not None')
    return input_


def number(input_: Any,
           /
           ) -> TNum:
    """
    Validates an input to be a (non-complex) number.

    :param input_: input to be validated.
    :return: validated input of numeric type.
    :raises TypeError: if the input is not numeric.
    """
    if not check.is_numeric(input_):
        raise TypeError('value is not numeric')
    return input_


def integer(input_: Any,
            /
            ) -> int:
    """
    Validates an input to be an integer.

    :param input_: input to be validated.
    :return: validated input of int type.
    :raises TypeError: if the input is not an integer.
    """
    if not check.is_integer(input_):
        raise TypeError('value is not an integer')
    return input_


def less(number_: TNum,
         /,
         upper_limit: TNum = 0
         ) -> TNum:
    """
    Validates a numeric input to be less than an upper limit.

    :param number_: numeric input to be validated.
    :param upper_limit: upper limit below which the input is valid, default is 0.
    :return: validated input, a number less than upper_limit.
    :raises ValueError: if the number is greater than or equal to upper_limit;
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(upper_limit), 'upper_limit must be numeric'

    if not number_ < upper_limit:
        raise ValueError('value is not less than {}'.format(upper_limit))
    return number_


def less_equal(number_: TNum,
               /,
               upper_limit: TNum = 0
               ) -> TNum:
    """
    Validates a numeric input to be less than or equal to an upper limit.

    :param number_: numeric input to be validated.
    :param upper_limit: upper limit below and at which the input is valid, default is 0.
    :return: validated input, a number less than or equal to upper_limit.
    :raises ValueError: if the number is greater than upper_limit.
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(upper_limit), 'upper_limit must be numeric'

    if not number_ <= upper_limit:
        raise ValueError('value is not less than or equal to {}'.format(upper_limit))
    return number_


def equal(number_: TNum,
          /,
          comparison_value: TNum = 0
          ) -> TNum:
    """
    Validates a numeric input to be equal to a comparison value.

    :param number_: numeric input to be validated.
    :param comparison_value: value at which the input is valid, default is 0.
    :return: validated input, a value equal to comparison_value.
    :raises ValueError: if the value is not equal to comparison_value.
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(comparison_value), 'comparison_value must be numeric'
    if number_ != comparison_value:
        raise ValueError('value is not equal to {}'.format(comparison_value))
    return number_


def not_equal(number_: TNum,
              /,
              comparison_value: TNum = 0
              ) -> TNum:
    """
    Validates a numeric input to be not equal to a comparison value.

    :param number_: numeric input to be validated.
    :param comparison_value: value at which the input is invalid, default is 0.
    :return: validated input, a value not equal to comparison_value.
    :raises ValueError: if the value is equal to comparison_value.
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(comparison_value), 'comparison_value must be numeric'

    if number_ == comparison_value:
        raise ValueError('value is equal to {}'.format(comparison_value))
    return number_


def greater(number_: TNum,
            /,
            lower_limit: TNum = 0
            ) -> TNum:
    """
    Validates a numeric input to be greater than a lower limit.

    :param number_: numeric input to be validated.
    :param lower_limit: lower limit above which the input is valid, default is 0.
    :return: validated input, a number greater than lower_limit.
    :raises ValueError: if the number is less than or equal to lower_limit;
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(lower_limit), 'lower_limit must be numeric'

    if not number_ > lower_limit:
        raise ValueError('value is not greater than {}'.format(lower_limit))
    return number_


def greater_equal(number_: TNum,
                  /,
                  lower_limit: TNum = 0
                  ) -> TNum:
    """
    Validates a numeric input to be greater than or equal to a lower limit.

    :param number_: numeric input to be validated.
    :param lower_limit: lower limit below and at which the input is valid, default is 0.
    :return: validated input, a number greater than or equal to lower_limit.
    :raises ValueError: if the number is less than lower_limit.
    """
    assert check.is_numeric(number_), 'number_ must be numeric'
    assert check.is_numeric(lower_limit), 'lower_limit must be numeric'

    if not number_ >= lower_limit:
        raise ValueError('value is not greater than or equal to {}'.format(lower_limit))
    return number_


def sequence(input_: Any,
             /,
             sequence_type: None | type | list[type] = None,
             length: None | int | list[int] = None,
             type_: None | type | tuple[type] = None,
             member_validation_function: Callable[..., Any] | tuple[Callable[..., Any]] | None = None
             ) -> TSequence:
    """
    Validates an input to be a sequence with certain characteristics.

    :param input_: input to be validated.
    :param sequence_type: the type of sequence to be validated or a list of acceptable types, default is None indicating
                          no restriction.
    :param length: length of the sequence as int or list of acceptable length,
                   default is None indicating no restriction.
    :param type_: type of all member of the sequence or a tuple of types (must be the length of the sequence) indicating
                  the type of each element, default is None indicating no restriction.
    :param member_validation_function: a callable validation function (see _is_validation_function for specifications)
                                       to be applied to all members or a sequence of validation functions with same
                                       length as the tuple to be applied to the member at the same position,
                                       default is None indication no member validation by separate validation functions.
    :return: validated input, a sequence with given properties.
    :raises TypeError: if the input is not a sequence (of given type).
    :raises ValueError: if the sequence is not of given properties.
    """
    # assert parameters individually
    assert (isinstance(sequence_type, NoneType) or check.is_sequence_type(sequence_type)
            or isinstance(sequence_type, list) and all([check.is_sequence_type(sequence_type_)
                                                        for sequence_type_ in sequence_type])), \
        'sequence_type must be None, a sequence type or a list of sequence types'
    assert (isinstance(length, (int, NoneType))
            or isinstance(length, list) and all([isinstance(length_, int) for length_ in length])), \
        'length must be None, int or a list of int'
    assert (isinstance(type_, (NoneType, type))
            or isinstance(type_, tuple) and all([isinstance(member, type) for member in type_])), \
        'type_ must be None, type or a tuple of types'
    assert (isinstance(member_validation_function, NoneType) or _is_validation_function(member_validation_function)
            or isinstance(member_validation_function, tuple)
            and all([_is_validation_function(member_validation_function_)
                     for member_validation_function_ in member_validation_function])), \
        ('validation_functions must be callables with only one parameter without default value (positional-only). '
         'Consider binding other parameters with functools.partial or the wrapper called \'bind\' from this module.')

    # assert dependencies between parameters
    if isinstance(length, list):
        assert not isinstance(type_, tuple) and not isinstance(member_validation_function, tuple), \
            ('member-wise validation of type and/or by validation function does not work if multiple length are '
             'acceptable')
    if isinstance(type_, tuple) and length is not None:
        assert len(type_) == length, 'provided length must match type_, if provided as a tuple'
    if isinstance(member_validation_function, tuple) and length is not None:
        assert len(member_validation_function) == length, 'provided length must match type_, if provided as a tuple'
    if isinstance(type_, tuple) and isinstance(member_validation_function, tuple):
        assert len(type_) == len(member_validation_function), ('tuple length for member-wise validation of type '
                                                               'and by functions must match')

    if sequence_type is None:
        if not check.is_sequence(input_):
            raise TypeError('input is not a sequence')
    elif isinstance(sequence_type, type) and not isinstance(input_, sequence_type):
        raise TypeError('sequence is not a {}'.format(sequence_type))
    elif isinstance(sequence_type, list) and not any([isinstance(input_, sequence_type_)
                                                      for sequence_type_ in sequence_type]):
        raise TypeError('sequence is not any of {}'.format(sequence_type))
    if isinstance(length, int) and length != len(input_):
        raise ValueError('{} is not of length {}'
                         .format(type(input_).__name__ if sequence_type else 'sequence', length))

    if isinstance(length, list) and not len(input_) in length:
        raise ValueError('{} is not of any of the length in {}'
                         .format(type(input_).__name__ if sequence_type else 'sequence', length))
    if isinstance(type_, type) and not all([isinstance(member, type_) for member in input_]):
        raise ValueError('not all members of the {} are of type {}'
                         .format(type(input_).__name__ if sequence_type else 'sequence', type_))
    if isinstance(type_, tuple):
        if len(type_) != len(input_):
            raise ValueError('signature {} does not have matching length for the {}'
                             .format(type_, type(input_).__name__ if sequence_type else 'sequence',))
        if not all([isinstance(member, t) for member, t in zip(input_, type_)]):
            raise ValueError('{} does not have the type signature {}'
                             .format(type(input_).__name__ if sequence_type else 'sequence', type_))
    exceptions: list[Exception] = []
    if _is_validation_function(member_validation_function):
        for idx, member in enumerate(input_):
            try:
                member_validation_function(member, 'member at index {}'.format(idx))
            except Exception as e:
                exceptions.append(e)
    if isinstance(member_validation_function, tuple):
        for idx, (member, member_validation_function_) in enumerate(zip(input_, member_validation_function)):
            member_validation_function_(member, 'member at index {}'.format(idx))
    if exceptions:
        # TODO use ExceptionGroup once updated to py 3.11
        raise Exception('member-validation of {} failed. Exception(s) raised.'
                        .format(type(input_).__name__ if sequence_type else 'sequence'), exceptions)
    return input_


def tuple_(input_: Any,
           /,
           length: None | int | list[int] = None,
           type_: None | type | tuple[type] = None,
           member_validation_function: Callable[..., Any] | tuple[Callable[..., Any]] | None = None
           ) -> tuple:
    """
    Validates an input to be a tuple with certain properties.
    (Convenience wrapper for sequence validation function)

    :param input_: input to be validated.
    :param length: length of the tuple as int or list of acceptable length, default is None indicating no restriction.
    :param type_: type of all member of the tuple or a tuple of types (must be the length of the tuple) indicating the
                  type of each element, default is None indicating no restriction.
    :param member_validation_function: a callable validation function (see _is_validation_function for specifications)
                                       to be applied to all members or a sequence of validation functions with same
                                       length as the tuple to be applied to the member at the same position,
                                       default is None indication no member validation by separate validation functions.
    :return: validated input, a tuple with given properties.
    """
    return sequence(input_, tuple, length, type_, member_validation_function)


def list_(input_: Any,
          /,
          length: None | int | list[int] = None,
          type_: None | type | tuple[type] = None,
          member_validation_function: Callable[..., Any] | tuple[Callable[..., Any]] | None = None
          ) -> list:
    """
    Validates an input to be a list with certain properties.
    (Convenience wrapper for sequence validation function)

    :param input_: input to be validated.
    :param length: length of the list as int or list of acceptable length, default is None indicating no restriction.
    :param type_: type of all member of the list or a tuple of types (must be the length of the list) indicating the
                  type of each element, default is None indicating no restriction.
    :param member_validation_function: a callable validation function (see _is_validation_function for specifications)
                                       to be applied to all members or a sequence of validation functions with same
                                       length as the tuple to be applied to the member at the same position,
                                       default is None indication no member validation by separate validation functions.
    :return: validated input, a list with given properties.
    """
    return sequence(input_, list, length, type_, member_validation_function)


def numpy_array(input_: Any,
                /,
                shape: None | tuple[None | int, ...] | list[tuple[None | int, ...]] = None,
                min_value: None | TNum = None,
                max_value: None | TNum = None,
                dtype: None | type = None,
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
    :return: validated input, a numpy array of given shape and within given limits.
    :raises TypeError: if the input is not of proper shape, dtype is non-numeric or input cannot be cast to dtype.
    :raises ValueError: if the input array has (an) out of bounds value(s).
    """
    assert (isinstance(shape, NoneType) or check.is_numpy_array_shape(shape, allow_none=True)
            or isinstance(shape, list) and all([check.is_numpy_array_shape(shape_, allow_none=True)
                                                for shape_ in shape])), \
        'shape must be None, a tuple of int or a list of tuples of int'
    assert isinstance(min_value, NoneType) or check.is_numeric(min_value), 'min_value must be numeric or None'
    assert isinstance(max_value, NoneType) or check.is_numeric(max_value), 'max_value must be numeric or None'
    assert (isinstance(dtype, NoneType) or check.is_numpy_dtype(dtype)
            or isinstance(dtype, list) and all([check.is_numpy_dtype(dtype_) for dtype_ in dtype])), \
        'dtype must be None, a numeric type or a list of numeric types'

    array_ = np.array(input_)  # to capture compatible inputs
    if (isinstance(shape, tuple)
            and (len(shape) != len(input_.shape)
                 or not all([dim is None or dim == input_dim for dim, input_dim in zip(shape, input_.shape)]))):
        raise TypeError('numpy array (compatible) is not of shape {}'.format(shape).replace('None', 'Any'))
    if (isinstance(shape, list)
            and not any([len(shape_) == len(input_.shape)
                         and all([dim is None or dim == input_dim
                                  for dim, input_dim in zip(shape_, input_.shape)]) for shape_ in shape])):
        raise TypeError('numpy array (compatible) is not of any of the shapes in {}'
                        .format(shape).replace('None', 'Any'))
    if min_value is not None and np.min(array_) < min_value:
        raise ValueError('numpy array (compatible) contains at least one value below minimum {}'.format(min_value))
    if max_value is not None and np.max(array_) > max_value:
        raise ValueError('numpy array (compatible) contains at least one value above maximum {}'.format(max_value))
    if not np.can_cast(array_, dtype):
        raise TypeError('numpy array (compatible) is not castable to {}'.format(np.dtype(dtype).name))
    return array_.astype(dtype)
