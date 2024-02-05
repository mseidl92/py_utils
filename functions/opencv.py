"""
Classes:

- RGBColor: named RGB color values.
- HorizontalTextAlignment: enum of horizontal text alignment formats.
- VerticalTextAlignment: enum of vertical text alignment formats.
- WindowAnchor: enum of typical anchors of a window.

Functions.

- value_to_image_dtype: converts input to fit a given images data type.
- print_text_on_image: prints text on openCV image.

"""

__author__ = 'Marius Seidl'
__date__ = '2024-02-05'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import numpy as np
import warnings

# locally build library imports
import cv2

# project imports
from ..types_ import check
from ..exceptions.warnings_ import VisualizationWarning
from .helper_classes import ContainsEnum


class RGBColor:
    """
    Named RGB color values.
    Basic RBG colors from `here <https://www.rapidtables.com/web/color/RGB_Color.html>`.
    """
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    LIME = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    SILVER = (192, 192, 192)
    GRAY = (128, 128, 128)
    MAROON = (128, 0, 0)
    OLIVE = (128, 128, 0)
    GREEN = (0, 128, 0)
    PURPLE = (128, 0, 128)
    TEAL = (0, 128, 128)
    NAVY = (0, 0, 128)


class HorizontalTextAlignment(ContainsEnum):
    """
    Standard horizontal text alignment modes.
    """
    LEFT = 0
    CENTERED = 1
    RIGHT = 2
    JUSTIFIED = 3


class VerticalTextAlignment(ContainsEnum):
    """
    Standard vertical text alignment modes.
    """
    TOP = 0
    CENTERED = 1
    BOTTOM = 2
    SPREAD = 3


class WindowAnchor(ContainsEnum):
    """
    Standard anchor points for a rectangular window.

    x--------x--------x
    |                 |
    x        x        x
    |                 |
    x--------x--------x

    """
    TOP_LEFT = 0
    LEFT_TOP = 0

    CENTER_LEFT = 1
    LEFT_CENTER = 1

    BOTTOM_LEFT = 2
    LEFT_BOTTOM = 2

    TOP_CENTER = 3
    CENTER_TOP = 3

    CENTER = 4
    CENTER_CENTER = 4

    BOTTOM_CENTER = 5
    CENTER_BOTTOM = 5

    TOP_RIGHT = 6
    RIGHT_TOP = 6

    CENTER_RIGHT = 7
    RIGHT_CENTER = 7

    BOTTOM_RIGHT = 8
    RIGHT_BOTTOM = 8


def value_to_image_dtype(value: int | float | np.ndarray,
                         target: np.ndarray | np.dtype
                         ) -> int | float | np.ndarray:
    """
    Converts an input value to a target dtype for an openCV image.
    Shifts and rescales value as required to conserve representation of the input.

    :param value: Value to convert. Can be an integer in the unsigned 8-bit range, a float in the interval [0., .1] or
    a numpy array representing a valid openCV image.
    :param target: Target of the conversation. Can be a valid openCV image which type becomes the target type or a
    directly provided numpy dtype.
    :return: The shifted, scaled and type-converted input value.
    """
    # validate input
    assert (isinstance(value, int) and 0 <= value <= 255 or isinstance(value, float) and 0. <= value <= 1.
            or isinstance(value, np.ndarray) and check.is_cv_image(value)), \
        'Expected value to be an int in [0, 255], a float in [0., 1.] or a valid openCV image as numpy array.'
    assert check.is_cv_image(target) or isinstance(target, np.dtype), \
        'Expected target to be a valid openCV image or numpy dtype object.'

    # unify input to dtype only
    target_dtype = target.dtype if isinstance(target, np.ndarray) else target

    if isinstance(value, int):
        if target_dtype in [np.int8, np.int16]:
            # shift to signed range
            value += np.iinfo(np.int8).min
        if target_dtype in [np.uint16, np.int16]:  # don't use else to treat int16 both ways
            # scale to 16-bit range
            value *= np.iinfo(np.uint16).max // np.iinfo(np.uint8).max
        elif target_dtype in [np.float16, np.float32]:  # use else here for efficiency
            # scale to unit range
            value /= np.iinfo(np.uint8).max
    elif isinstance(value, float):
        if target_dtype in [np.int8, np.int16]:
            # shift to signed range
            value -= .5
        if target_dtype in [np.uint8, np.int8]:  # don't use else to treat int8 and int16 both ways
            # scale to 8-bit range
            value = int(value * np.iinfo(np.uint8).max)
        elif target_dtype in [np.uint16, np.int16]:  # use else here for efficiency
            # scale to 16-bit range
            value = int(value * np.iinfo(np.uint8).max)
    elif isinstance(value, np.ndarray):
        # convert value dtype to allow for operations outside range
        value_dtype = value.dtype
        value = value.astype(np.float64)
        # shift sign if required
        if target_dtype in [np.uint8, np.uint16, np.float32, np.float64] and value_dtype in [np.int8, np.int16]:
            # shift to unsigned range if original value is signed
            value -= np.iinfo(value_dtype).min
        elif target_dtype in [np.int8, np.int16] and value_dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            # shift to signed range if original value was unsigned
            if value_dtype == np.uint8:
                value += np.iinfo(np.int8).min
            elif value_dtype == np.uint16:
                value += np.iinfo(np.int16).min
            elif value_dtype in [np.float32, np.float64]:
                value -= .5
        # shift scale if required
        if target_dtype in [np.uint8, np.int8]:
            if value_dtype in [np.uint16, np.int16]:
                # scale to 16-bit range
                value *= np.iinfo(np.uint16).max // np.iinfo(np.uint8).max
            elif value_dtype in [np.float32, np.float64]:
                # scale to unit range
                value /= np.iinfo(np.int8).max
        elif target_dtype in [np.uint16, np.int16]:
            if value.dtype in [np.uint8, np.int8]:
                # scale to 8-bit range
                value /= np.iinfo(np.uint16).max // np.iinfo(np.uint8).max
            elif value_dtype in [np.float32, np.float64]:
                # scale to unit range
                value /= np.iinfo(np.int16).max
        elif value_dtype in [np.float32, np.float64]:
            if value.dtype in [np.uint8, np.int8]:
                # scale to 8-bit range
                value *= np.iinfo(np.uint8).max
            elif value_dtype in [np.uint16, np.int16]:
                # scale to 16-bit range
                value *= np.iinfo(np.uint16).max
        # covert to desired output type
        value = value.astype(target_dtype)

    return value


def print_text_on_image(text: tuple[str] | str,
                        image: np.ndarray,
                        text_area: tuple[int, int, int, int],
                        default_font_scale: int | float = 1,
                        default_border_width: int = 20,
                        default_line_distance: int = 5,
                        default_font_thickness: int = 2,
                        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
                        font_color: tuple[int, int, int] = RGBColor.BLACK,
                        font_alpha: int | float = 1,
                        horizontal_alignment: int | HorizontalTextAlignment = HorizontalTextAlignment.LEFT,
                        vertical_alignment: int | VerticalTextAlignment = VerticalTextAlignment.TOP,
                        rotation_degrees: int | float = 0,
                        rotation_anchor: int | WindowAnchor = WindowAnchor.TOP_LEFT
                        ) -> np.ndarray:
    # TODO debug function
    """
    Print text onto an openCV image. The text will be printed with default_font_scale or scaled down to fit the
    text_area if it is too large at the default scale.

    :param text: text string to print with multiple lines as tuple of strings.
    :param image: the original image to print the text onto as numpy array.
    :param text_area: tuple of ints indicating the text area as (x, y, width, height),
    with (x,y) being the top-left corner.
    :param default_font_scale: maximum font scale to print at as float.
    :param default_border_width: border width at default_font_scale in px as int.
    :param default_line_distance: line distance at default_font_scale in px as int.
    :param default_font_thickness: font thickness at default_font_scale in px as int
    :param font_face: openCV font face.
    :param font_color: font color as RGB tuple of ints in [0, 255].
    :param font_alpha: font alpha as float in [0, 1]. Ignored if input image has no alpha channel.
    :param horizontal_alignment: horizontal alignment of the text: LEFT, CENTER, RIGHT, JUSTIFIED.
    :param vertical_alignment: vertical alignment of the text: TOP, CENTER, BOTTOM, SPREAD.
    :param rotation_degrees: rotation of the text box in degree around the rotation_anchor.
    :param rotation_anchor: position around which the box is rotated: TOP_LEFT, TOP_CENTER, TOP_RIGHT, CENTER_LEFT,
    CENTER, CENTER_RIGHT, BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT.
    :return: the image with text printed on it as numpy array.
    """

    # validate inputs
    assert isinstance(text, str) or (isinstance(text, tuple) and all([isinstance(line, str) for line in text])), \
        'Expected type of text to be string or tuple of string.'
    assert check.is_cv_image(image), 'Expected image to be a valid openCV image numpy array.'
    assert (isinstance(text_area, tuple) and len(text_area) == 4 and all([isinstance(ta, int) for ta in text_area])
            and 0 <= text_area[0] <= image.shape[1] and 0 <= text_area[1] <= image.shape[0]
            and 0 < text_area[2] <= image.shape[1] and 0 < text_area[3] <= image.shape[0]), \
        ('Expected text_area to be a tuple of four ints representing (x, y, width, height) '
         'within the bounds of the original image.')
    assert isinstance(default_font_scale, (int, float)) and default_font_scale > 0., \
        'Expected default_font_scale to be an int or float > 0.'
    assert isinstance(default_border_width, int) and default_border_width >= 1, \
        'Expected default_border_width to be a int >= 1.'
    assert isinstance(default_line_distance, int) and default_line_distance >= 1, \
        'Expected default_line_distance to be a int >= 1'
    assert isinstance(default_font_thickness, int) and default_font_thickness >= 1, \
        'Expected default_font_thickness to be a int >= 1'
    assert (isinstance(font_face, int)
            and font_face in [i for i in range(8)] + [i | cv2.FONT_ITALIC for i in range(8)]), \
        'Expected font_face to be a int representing a valid openCV font face type.'
    assert (isinstance(font_color, tuple) and len(font_color) == 3
            and all([isinstance(value, int) and 0 <= value <= 255 for value in font_color])), \
        'Expected font_color to be a RGB color tuple with int values in the [0, 255] interval.'
    assert isinstance(font_alpha, (int, float)) and 0. <= font_alpha <= 1., \
        'Expected font_alpha to be an int or float in the [0., 1.] interval.'
    assert (isinstance(horizontal_alignment, (int, HorizontalTextAlignment))
            and horizontal_alignment in HorizontalTextAlignment), \
        'Expected horizontal_alignment to be a int representing a valid HorizontalTextAlignment enum value.'
    assert (isinstance(vertical_alignment, (int, VerticalTextAlignment))
            and vertical_alignment in VerticalTextAlignment), \
        'Expected vertical_alignment to be a int representing a valid VerticalTextAlignment enum value.'
    assert isinstance(rotation_degrees, (int, float)), 'Expected rotation_degree to be a int or float'
    assert isinstance(rotation_anchor, (int, WindowAnchor)) and rotation_anchor in WindowAnchor, \
        'Expected rotation_anchor to be a int representing a valid WindowAnchor enum value.'

    # define some variable types_
    font_scale: float
    border_width: int
    line_distance: int
    font_thickness: int

    n_lines: int
    line_width_max: int
    line_height_max: int

    # unpack inputs
    text_area_origin_x, text_area_origin_y, text_area_width, text_area_height = text_area

    # get text format
    if isinstance(text, str):
        text = (text,)
    n_lines = len(text)
    line_width_max = 0
    line_height_max = 0
    for line in text:
        ((line_width, line_height), _) = cv2.getTextSize(line, font_face, default_font_scale, default_font_thickness)
        if line_width > line_width_max:
            line_width_max = line_width
        if line_height > line_height_max:
            line_height_max = line_height

    # get scale
    if (text_area_width < line_width_max + 2 * default_border_width * default_font_scale
            or text_area_height < (n_lines * line_height_max + (n_lines + 1)
                                   * default_line_distance * default_font_scale)):
        # solve font_scale = text_area_width / (line_width_max + 2 * font_border_unscale * font_scale)
        font_scale_width_lim = (np.sqrt((line_width_max / (4 * default_border_width)) ** 2
                                        + text_area_width / (2 * default_border_width))
                                - line_width_max / (4 * default_border_width))
        # guarantee at least 1px border width
        if int(default_border_width * font_scale_width_lim) < 1:
            font_scale_width_lim = text_area_width / (line_width_max + 2)
        # solve font_scale = text_area_height / (n_lines * line_height_max
        #                                        + (n_lines + 1) * (default_line_distance * font_scale)))
        font_scale_height_lim = (np.sqrt((n_lines * line_height_max / (2 * (n_lines + 1) * default_line_distance)) ** 2
                                         + text_area_height / ((n_lines + 1) * default_line_distance))
                                 - n_lines * line_height_max / (2 * (n_lines + 1) * default_line_distance))
        # guarantee at least 1px line distance
        if int(default_line_distance * font_scale_width_lim) < 1:
            font_scale_height_lim = text_area_height / (n_lines * text_area_height + n_lines + 1)
        # take minimum to definitely fit
        font_scale = min(font_scale_height_lim, font_scale_width_lim)
        border_width = max(1, int(default_border_width * font_scale))
        line_distance = max(1, int(default_line_distance * font_scale))
        font_thickness = max(1, int(default_font_thickness * font_scale))
    else:
        font_scale = default_font_scale
        border_width = default_border_width
        line_distance = default_line_distance
        font_thickness = default_font_thickness

    # calculate text origins for openCV putText function
    x_origins: list[int | list[int]] = []
    match horizontal_alignment:
        case HorizontalTextAlignment.LEFT:
            x_origins = [border_width] * len(text)
        case HorizontalTextAlignment.CENTERED:
            for line in text:
                ((line_width, _), _) = cv2.getTextSize(line, font_face, font_scale, font_thickness)

                x_origins.append((text_area_width - line_width) // 2)
        case HorizontalTextAlignment.RIGHT:
            for line in text:
                ((line_width, _), _) = cv2.getTextSize(line, font_face, font_scale, font_thickness)
                x_origins.append(text_area_width - line_width)
        case HorizontalTextAlignment.JUSTIFIED:
            x_origins_line: list[int]
            for line in text:
                word_widths: list[int] = []
                word_width_sum: int = 0
                for word in line.split():
                    ((word_width, _), _) = cv2.getTextSize(word, font_face, font_scale, font_thickness)
                    word_widths.append(word_width)
                    word_width_sum += word_width
                x_origins_line = []
                if len(word_widths) >= 1:
                    x_origins_line.append(border_width)
                if len(word_widths) >= 2:
                    space_between_words = ((text_area_width - word_width_sum - 2 * border_width)
                                           // (len(word_widths) - 1))
                    for word_width in word_widths[:-1]:
                        x_origins_line.append(x_origins_line[-1] + word_width + space_between_words)
                x_origins.append(x_origins_line)

    y_origins: list[int] = []
    # can be used for all cases
    height_sum = 0
    for line in text:
        ((_, line_height), _) = cv2.getTextSize(line, font_face, font_scale, font_thickness)
        height_sum += line_height + line_distance
        y_origins.append(height_sum)
    height_sum += line_distance
    # move the complete arrangement further down to match requested alignment
    match vertical_alignment:
        case VerticalTextAlignment.CENTERED:
            y_origins = [y_origin + (text_area_height - height_sum) // 2 for y_origin in y_origins]
        case VerticalTextAlignment.BOTTOM:
            y_origins = [y_origin + text_area_height - height_sum for y_origin in y_origins]
        case VerticalTextAlignment.SPREAD:
            if n_lines > 1:
                height_padding = (text_area_height - height_sum) // (n_lines-1)
                for n_line in range(n_lines):
                    y_origins[n_line] += height_padding * n_line

    # put text on image
    # create separate text image as mask
    # use padding to prevent rotating out of the frame
    if text_area_width > text_area_height:
        text_square_size = text_area_width
        x_padding = 0
        y_padding = (text_area_width - text_area_height) // 2
    else:
        text_square_size = text_area_height
        x_padding = (text_area_height - text_area_width) // 2
        y_padding = 0
    rotation_padding = int(np.ceil(text_square_size * (np.sqrt(2) - 1) / 2))
    text_image_mask = np.zeros((text_square_size + 2 * rotation_padding, text_square_size + 2 * rotation_padding),
                               dtype='uint8')
    for line, x_origin, y_origin in zip(text, x_origins, y_origins):
        if horizontal_alignment == HorizontalTextAlignment.JUSTIFIED:
            for word, x_origin_ in zip(line.split(), x_origin):
                text_image_mask = cv2.putText(text_image_mask, word, (x_origin_ + x_padding + rotation_padding,
                                                                      y_origin + y_padding + rotation_padding),
                                              font_face, font_scale, 1, font_thickness)
        else:
            text_image_mask = cv2.putText(text_image_mask, line, (x_origin + x_padding + rotation_padding,
                                                                  y_origin + y_padding + rotation_padding),
                                          font_face, font_scale, 1, font_thickness)
    # rotate mask around center
    rotation_matrix = cv2.getRotationMatrix2D((text_square_size // 2 + rotation_padding,
                                               text_square_size // 2 + rotation_padding), rotation_degrees, 1)
    text_image_mask = cv2.warpAffine(text_image_mask, rotation_matrix,
                                     (text_image_mask.shape[1], text_image_mask.shape[0]), cv2.INTER_AREA)
    cv2.imshow('test', text_image_mask * 255)
    text_image_mask = text_image_mask.astype(bool)

    # calculate origin of image overlay based on rotation anchor, since rotation was actually around the center
    rotation_center_x = text_square_size // 2 + rotation_padding
    rotation_center_y = text_square_size // 2 + rotation_padding
    if rotation_anchor in [WindowAnchor.TOP_LEFT, WindowAnchor.CENTER_LEFT, WindowAnchor.BOTTOM_LEFT]:
        rotation_center_x -= text_area_width // 2
    elif rotation_anchor in [WindowAnchor.TOP_RIGHT, WindowAnchor.CENTER_RIGHT, WindowAnchor.BOTTOM_RIGHT]:
        rotation_center_x += text_area_width // 2
    if rotation_anchor in [WindowAnchor.TOP_LEFT, WindowAnchor.TOP_CENTER, WindowAnchor.TOP_RIGHT]:
        rotation_center_y -= text_area_height // 2
    elif rotation_anchor in [WindowAnchor.BOTTOM_LEFT, WindowAnchor.BOTTOM_CENTER, WindowAnchor.BOTTOM_RIGHT]:
        rotation_center_y += text_area_height

    c, s = np.cos(np.deg2rad(rotation_degrees)), np.sin(np.deg2rad(rotation_degrees))
    dx_pre_rotation = rotation_center_x - (text_square_size // 2 + rotation_padding)
    dy_pre_rotation = rotation_center_y - (text_square_size // 2 + rotation_padding)
    # use negative sign and transpose of rotation matrix since motion is tracked backwards
    dx = (-int(dx_pre_rotation - c * dx_pre_rotation - s * dy_pre_rotation)
          - rotation_padding - x_padding + text_area_origin_x)
    dy = (-int(dy_pre_rotation + s * dx_pre_rotation - c * dy_pre_rotation)
          - rotation_padding - y_padding + text_area_origin_y)
    # dx and dy are swapped here because openCV follows image.shape[0:2] = (height, width) convention
    dx, dy = dy, dx

    # combine mask with original image based on its type
    x_lb_mask = max(0, -dx)
    x_ub_mask = min(image.shape[0] - dx, text_image_mask.shape[0])
    y_lb_mask = max(0, -dy)
    y_ub_mask = min(image.shape[1] - dy, text_image_mask.shape[1])

    x_lb_image = max(0, dx)
    x_ub_image = min(image.shape[0], dx + text_image_mask.shape[0])
    y_lb_image = max(0, dy)
    y_ub_image = min(image.shape[1], dy + text_image_mask.shape[1])

    # check if not used part of text image is empty, warn if something is cut and lost
    if (np.sum(text_image_mask[:x_lb_mask, :]) + np.sum(text_image_mask[x_ub_mask:, :])
       + np.sum(text_image_mask[:, :y_lb_mask]) + np.sum(text_image_mask[:, y_ub_mask:])) > 0:
        warnings.warn('Some or all of the text to be displayed is outside of image bounds.', VisualizationWarning)

    text_image_mask = text_image_mask[x_lb_mask:x_ub_mask, y_lb_mask:y_ub_mask]
    if len(image.shape) == 2:  # grayscale image
        # get gray value of rgb color
        gray_value = int(0.299 * font_color[0] + 0.587 * font_color[1] + 0.114 * font_color[2])
        sub_image = image[x_lb_image:x_ub_image, y_lb_image:y_ub_image]
        sub_image[text_image_mask] = value_to_image_dtype(gray_value, image)
        image[x_lb_image:x_ub_image, y_lb_image:y_ub_image] = sub_image
    elif image.shape[2] == 3:  # bgr image
        sub_image = image[x_lb_image:x_ub_image, y_lb_image:y_ub_image, :]
        sub_image[text_image_mask, 0] = value_to_image_dtype(font_color[2], image)
        sub_image[text_image_mask, 1] = value_to_image_dtype(font_color[1], image)
        sub_image[text_image_mask, 2] = value_to_image_dtype(font_color[0], image)
        image[x_lb_image:x_ub_image, y_lb_image:y_ub_image, :] = sub_image
    elif image.shape[2] == 4:  # bgr + alpha image:
        # use alpha compositing
        # https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
        # make sure to operate on the same [0., 1.] alpha interval for the input alpha value and the image
        sub_image = image[x_lb_image:x_ub_image, y_lb_image:y_ub_image, :]
        sub_image = value_to_image_dtype(sub_image, np.float64)
        bg_alpha = sub_image[:, :, 3]
        fg_alpha = text_image_mask.astype(np.float64) * font_alpha
        sub_image[:, :, 0] = fg_alpha * font_color[2] / 255 + bg_alpha * sub_image[:, :, 0] * (1 - fg_alpha)
        sub_image[:, :, 1] = fg_alpha * font_color[1] / 255 + bg_alpha * sub_image[:, :, 1] * (1 - fg_alpha)
        sub_image[:, :, 2] = fg_alpha * font_color[0] / 255 + bg_alpha * sub_image[:, :, 2] * (1 - fg_alpha)
        sub_image[:, :, 3] = (1 - (1 - fg_alpha) * (1 - bg_alpha))
        # restore original image dtype
        sub_image = value_to_image_dtype(sub_image, image.dtype)
        image[x_lb_image:x_ub_image, y_lb_image:y_ub_image, :] = sub_image

    return image
