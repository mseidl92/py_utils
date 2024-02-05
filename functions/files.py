"""
Functions:

- load_json: json file loading helper.

"""

__author__ = 'Marius Seidl'
__date__ = '2024-02-05'
__version__ = '1.0'
__license__ = 'GPL-3.0-or-later'

# standard library imports
import json


def load_json(file_path: str
              ) -> dict:
    """
    Loads json file.

    :param file_path: path to the json file.
    :return: dict containing json file content
    """
    # TODO make type save and handle exceptions
    with open(file_path) as fp:
        return json.load(fp)
