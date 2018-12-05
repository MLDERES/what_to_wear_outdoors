import os

DATA_PATH = os.path.relpath('../data')
MODEL_PATH = os.path.relpath('../models')


# TODO: Fix this up so we are more assurred of where the models are coming from
def get_model_name(athlete, sport, item):
    return '_'.join([athlete, sport, item]) + '.mdl'


def get_model_filename(item, athlete='michael', sport='run'):
    return os.path.join(MODEL_PATH, get_model_name(athlete, sport, item))
