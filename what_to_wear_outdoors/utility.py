from os import path
from pathlib import Path
import datetime as dt

from what_to_wear_outdoors.config import training_data_filename, test_data_filename

_ROOT = Path(__file__).parent


def get_data_path(filename='.') -> Path:
    """ Get the full path to a file in the data directory

    :param filename: name of the file for which to get the full path
    :return: a Path object that points to the file passed in the filename argument
    """
    return _ROOT / 'data' / filename


def get_model_path(filename) -> Path:
    """ Get the full path to a file in the model directory

        :param filename: name of the file for which to get the full path
        :return: a Path object that points to the file passed in the filename argument
    """
    return _ROOT / 'models' / filename


def get_model_name(sport, cookie='', athlete='default') -> str:
    """
    Creates the name of the model based on the athlete and sport.  This function is to ensure consistency
    when creating or opening models
    :type cookie: str
    :param cookie: any distinguishing aspect needed to create the model beside sport or athlete.  Can be None
    :type athlete: str
    :param athlete: athlete identifier for the specific model
    :type sport: str
    :param sport: sport for  the model name
    :return: a filename corresponding to the item, athlete and sport
    """
    return '_'.join([athlete, sport, cookie]) + '.mdl'


def get_boolean_model(sport, athlete='default') -> Path:
    """
    Return the model name for the Boolean models
    :param athlete:athlete identifier for the specific model
    :param sport: sport for  the model name
    :return: a path to the boolean model name
    """
    return get_model_path(get_model_name(sport, 'bool', athlete))


def get_categorical_model(sport, athlete='default') -> Path:
    """
    Return the model name for the categorical models
    :param athlete:athlete identifier for the specific model
    :param sport: sport for  the model name
    :return: a path to the categorical model name
    """
    return get_model_path(get_model_name(sport, 'cat', athlete))


def get_training_data_path(sport: str = '') -> Path:
    """ Path to the file with known good source data"""
    return get_data_path(training_data_filename)


def get_test_data_path() -> Path:
    """ Path to the XSLX file with good test data (not to be used for training. """
    return get_data_path(test_data_filename)


def is_file_newer(fn, days=0, hours=0, minutes=0):
    """ Check if a file has been modified within a given time """
    td = dt.timedelta(days=days, hours=hours, minutes=minutes)
    f = Path(fn)
    now = dt.datetime.now()
    return (now - dt.datetime.fromtimestamp(path.getmtime(f))) <= td


def file_exists(fn):
    f = Path(fn)
    return f.exists()


def read_int(key):
    try:
        return None if int(key) <= -999 else int(key)
    except ValueError:
        return None


def read_float(key):
    try:
        return None if float(key) <= -999 else float(key)
    except ValueError:
        return None
