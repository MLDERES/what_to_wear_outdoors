"""
In this module we are going to test the big scenarios from start to finish
"""
from pytest import fixture

from what_to_wear_outdoors import RunningOutfitPredictor
import pandas as pd


@fixture
def one_forecast():
    return {'feels_like': 69, 'duration': 45, 'temp_f': 69, 'wind_dir': 'E', 'condition': 'Clear',
            'distance': 4, 'activity': 'Run', 'wind_speed': 15, 'is_light': True, 'pct_humidity': 20}


@fixture
def forecasts():
    return {'feels_like': [69, 69], 'duration': [40, 45], 'temp_f': [69, 69], 'wind_dir': ['E', 'WSW'],
            'condition': ['Clear', 'Mostly Cloudy'], 'distance': [4, 4.2], 'activity': ['run', 'run'],
            'wind_speed': [0, 15], 'is_light': [True, False], 'pct_humidity': [20, 80]}


def results():
    return pd.DataFrame({'arm_warmers': [], 'base_layer': [], 'ears_hat': [], 'face_cover': [], 'gloves': [],
                         'heavy_socks': [], 'jacket': [], 'lower_body': [], 'outer_layer': [], 'shoe_cover': []})


def test_predict_outfit(one_forecast):
    """
    Send one forecast to the predictor model and see if it can correctly predict the outcome
    :return:
    """
    rop = RunningOutfitPredictor()
    print(f'{rop._supported_features}')
    fct = one_forecast
    rop.predict_outfit(**fct)
    pass


def test_predict_outfits():
    """
    Send a set of forecasts and we want to get back a set of outfits
    Passes the test if we get back one prediction for each outfit we send
    """
    pass


def test_score_predictor_model():
    """
    Using the actual data, score the model.
    Passes if we get a good score on the model (not sure what good is yet)
    :return:
    """
    pass
