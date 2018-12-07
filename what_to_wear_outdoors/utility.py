from pathlib import Path

_ROOT = Path(__file__).parent


def get_data(path):
    return _ROOT / 'data' / path


def get_model(path):
    return _ROOT / 'models' / path


def get_model_name(item, athlete='default', sport='run'):
    return '_'.join([athlete, sport, item]) + '.mdl'
