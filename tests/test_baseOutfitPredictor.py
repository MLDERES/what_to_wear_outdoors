import os
import random
from pathlib import Path
from pprint import pprint
from unittest import TestCase
import pytest
import logging
from pytest import mark, fixture
import pandas as pd
from what_to_wear_outdoors import utility, Weather, FctKeys, Features, get_data_path, get_test_data_path, \
    get_training_data_path
from what_to_wear_outdoors.outfit_predictors import BaseOutfitPredictor, RunningOutfitPredictor
import datetime as dt

TODAY = dt.date.today()
NOW = dt.datetime.now()

# Setup logging of debug messages to go to the file debug.log and the INFO messages
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='debug.log',
                    filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logging.getLogger('').addHandler(ch)

def temp_file_path(filename):
    _ROOT = Path(Path(__file__).anchor)
    return _ROOT / 'temp' / filename


@fixture
def build_temp_data():
    rop = RunningOutfitPredictor()
    f = Weather.random_forecast()
    outfit = {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
              'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
              'heavy_socks': False, 'arm_warmers': True, 'face_cover': False}
    duration = max(20, round(random.normalvariate(45, 45)))
    distance = round((duration / random.triangular(8, 15, 10.5)), 2)
    d = max(20, random.normalvariate(45, 45))
    df = rop.add_to_sample_data(f, outfit=outfit, athlete_name='Jim', activity_date=NOW, duration=d, distance=distance)
    df = rop.add_to_sample_data(vars(f), outfit=outfit, athlete_name='Default', activity_date=NOW, duration=d,
                                distance=distance)
    return rop.write_sample_data()


@fixture
def dataframe_format():
    rop = RunningOutfitPredictor()
    df = rop.get_dataframe_format()
    return df


@fixture
def predictor():
    return RunningOutfitPredictor()


@mark.parametrize("filename",
                  [
                      'all',
                      get_training_data_path(),
                      pytest.param('', marks=[mark.xfail]),
                      pytest.param(get_test_data_path()),
                  ],
                  )
@mark.parametrize('include_xl', [True, False])
def test_ingest_data(build_temp_data, dataframe_format, predictor, filename, include_xl):
    p = predictor
    df = p.ingest_data(filename, include_xl)
    c = list(df.columns)
    d = list(dataframe_format.columns)
    c.sort()
    d.sort()
    assert c == d


@fixture
def layers_df():
    return pd.DataFrame({'jacket': ['Wind', 'Wind', 'Wind', 'None', 'None', 'None', 'None'],
                         'outer_layer': ['A', 'A', 'None', 'A', 'A', 'None', 'None'],
                         'base_layer': ['B', 'None', 'B', 'B', 'None', 'B', 'None']})


def test__fix_layers(predictor, layers_df):
    ex_df = pd.DataFrame({'jacket': ['Wind', 'Wind', 'Wind', 'None', 'None', 'None', ],
                          'outer_layer': ['A', 'None', 'None', 'A', 'A', 'B', ],
                          'base_layer': ['B', 'A', 'B', 'B', 'None', 'None', ]})

    outcome = predictor._fix_layers(layers_df)
    assert ex_df.equals(outcome)


class TestBaseOutfitPredictor(TestCase):

    def test_predictors(self):
        bop = BaseOutfitPredictor()
        p = bop.features
        self.assertListEqual(
            [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY, Features.DURATION, Features.LIGHT], p)

    def test_predict_outfit_mild_no_light(self):
        """
        Testing the prediction for running outfits when it's not light out.
        :return:
        """
        rop = RunningOutfitPredictor()
        predicted_outfit = rop.predict_outfit(**{FctKeys.FEEL_TEMP: 50, FctKeys.WIND_SPEED: 5, FctKeys.HUMIDITY: .80,
                                                 Features.DURATION: 30, Features.LIGHT: True})
        self.assertDictEqual(predicted_outfit,
                             {'outer_layer': 'Long-sleeve', 'base_layer': 'None', 'jacket': 'None',
                              'lower_body': 'Shorts-calf cover', 'shoe_cover': 'None', 'ears_hat': False,
                              'gloves': False, 'heavy_socks': False, 'arm_warmers': False, 'face_cover': False})

    def test_predict_outfit_mild_light(self):
        """
        Testing the prediction for running outfits when there is still sunlight.
        :return:
        """
        rop = RunningOutfitPredictor()
        self.assertDictEqual(
            rop.predict_outfit(**{FctKeys.FEEL_TEMP: 54, FctKeys.WIND_SPEED: 15, FctKeys.HUMIDITY: .58,
                                  Features.DURATION: 115, Features.LIGHT: False}),
            {'outer_layer': 'Long-sleeve', 'base_layer': 'None', 'jacket': 'None',
             'lower_body': 'Shorts-calf cover', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
             'heavy_socks': False, 'arm_warmers': False, 'face_cover': False})

    def test_predict_outfit_warm_light(self):
        """
        Testing the prediction for running outfits when it's not light out.
        :return:
        """
        rop = RunningOutfitPredictor()
        self.assertDictEqual(
            rop.predict_outfit(
                **{FctKeys.FEEL_TEMP: 61, FctKeys.WIND_SPEED: 0, FctKeys.HUMIDITY: .42, Features.DURATION: 31,
                   Features.LIGHT: True}),
            {'outer_layer': 'Long-sleeve', 'base_layer': 'None', 'jacket': 'None',
             'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
             'heavy_socks': True, 'arm_warmers': False, 'face_cover': False})

    def test_predict_outfit_warm_short_sleeve(self):
        """
        Testing the prediction for running outfits when it's not light out.
        :return:
        """
        rop = RunningOutfitPredictor()
        self.assertDictEqual(
            rop.predict_outfit(
                **{FctKeys.FEEL_TEMP: 72, FctKeys.WIND_SPEED: 9, FctKeys.HUMIDITY: .38, Features.DURATION: 55,
                   Features.LIGHT: True}),
            {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
             'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
             'heavy_socks': False, 'arm_warmers': False, 'face_cover': False})

    def test_add_to_sample_data(self):
        rop = RunningOutfitPredictor()
        f = Weather.random_forecast()
        outfit = {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
                  'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
                  'heavy_socks': False, 'arm_warmers': True, 'face_cover': False}
        duration = max(20, round(random.normalvariate(45, 45)))
        distance = round((duration / random.triangular(8, 15, 10.5)), 2)
        d = max(20, random.normalvariate(45, 45))
        df = rop.add_to_sample_data(f, outfit=outfit, athlete_name='Jim', duration=d, distance=distance)
        pprint(df.to_string())
        df = rop.add_to_sample_data(vars(f), outfit=outfit, athlete_name='Default', duration=d, distance=distance)
        pprint(df.to_string())

    def test_write_sample_data(self):
        rop = RunningOutfitPredictor()
        f = Weather.random_forecast()
        outfit = {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
                  'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
                  'heavy_socks': False, 'arm_warmers': True, 'face_cover': False}
        duration = max(20, round(random.normalvariate(45, 45)))
        distance = round((duration / random.triangular(8, 15, 10.5)), 2)
        d = max(20, random.normalvariate(45, 45))
        df = rop.add_to_sample_data(f, outfit=outfit, athlete_name='Jim', duration=d, distance=distance)
        pprint(df.to_string())
        df = rop.add_to_sample_data(vars(f), outfit=outfit, athlete_name='Default', duration=d, distance=distance)
        pprint(df.to_string())
        rop.write_sample_data()

    def test_get_dataframe_format(self):
        bop = BaseOutfitPredictor()
        df = bop.get_dataframe_format()
        print(df.dtypes)
        print(df)

    def test_ingest_data_test_and_training(self):
        rop = RunningOutfitPredictor()
        df1 = rop.ingest_data('training_data.csv', include_xl=False)
        df2 = rop.ingest_data('what i wore running.xlsx', include_xl=False)
        df1_cols = list(df1.columns)
        df2_cols = list(df2.columns)
        df1_cols.sort()
        df2_cols.sort()
        assert df1_cols == df2_cols
