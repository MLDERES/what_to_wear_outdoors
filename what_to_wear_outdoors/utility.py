from pathlib import Path

_ROOT = Path(__file__).parent


def get_data_path(filename) -> object:
    """ Get the full path to a file in the data directory

    :param filename: name of the file for which to get the full path
    :return: a Path object that points to the file passed in the filename argument
    """
    return _ROOT / 'data' / filename


def get_model(filename):
    """ Get the full path to a file in the model directory

        :param filename: name of the file for which to get the full path
        :return: a Path object that points to the file passed in the filename argument
    """
    return _ROOT / 'models' / filename


def get_model_name(prefix, athlete='default', sport='run'):
    """
    Creates the name of the model based on the athlete and sport.  This function is to ensure consistency
    when creating or opening models
    :param prefix: name of the clothing item for which the model represents
    :param athlete: athlete identifier for the specific model
    :param sport: sport for  the model name
    :return: a filename corresponding to the item, athlete and sport
    """
    return '_'.join([athlete, sport, prefix]) + '.mdl'

def get_boolean_model(athlete='default', sport='run') -> Path:
    """
    Return the model name for the Boolean models
    :param athlete:athlete identifier for the specific model
    :param sport: sport for  the model name
    :return: a path to the boolean model name
    """
    return get_model(get_model_name('bool', athlete, sport))

def get_categorical_model(athlete='default', sport='run') -> Path:
    """
    Return the model name for the categorical models
    :param athlete:athlete identifier for the specific model
    :param sport: sport for  the model name
    :return: a path to the categorical model name
    """
    return get_model(get_model_name('cat',athlete,sport))



