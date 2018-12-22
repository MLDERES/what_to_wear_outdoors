import os
import random
from pathlib import Path
from pprint import pprint
from unittest import TestCase
import pytest
from pytest import mark, fixture
import pandas as pd
from what_to_wear_outdoors import utility, Weather, FctKeys
from what_to_wear_outdoors.outfit_predictors import BaseOutfitPredictor, RunningOutfitPredictor, Features
import datetime as dt

TODAY = dt.date.today()
NOW = dt.datetime.now()


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
    return rop.write_sample_data('test_outfit.csv')


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
                      'test_outfit.csv',
                      pytest.param('', marks=[mark.xfail]),
                      pytest.param('what i wore running.xlsx'),
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

    def test__are_models_outof_date(self):
        rop = RunningOutfitPredictor()
        cat_model_name = utility.get_categorical_model(sport=rop.activity_name)
        bool_model_name = utility.get_boolean_model(sport=rop.activity_name)
        Path.touch(utility.get_data_path('test_outfit.csv'))
        assert rop._are_models_out_of_date(cat_model_name, bool_model_name), 'Models are correctly out of date'
        Path.touch(cat_model_name)
        assert rop._are_models_out_of_date(cat_model_name, bool_model_name), 'cat model is new'
        Path.touch(bool_model_name)
        assert rop._are_models_out_of_date(cat_model_name, bool_model_name) is False, 'both models are new'

    def test_rebuild_models(self):
        """
        Retrain the models
        :return:
        """
        bop = RunningOutfitPredictor()
        if Path.exists(utility.get_categorical_model(sport=bop.activity_name)):
            os.remove(utility.get_categorical_model(sport=bop.activity_name))
        if Path.exists(utility.get_boolean_model(sport=bop.activity_name)):
            os.remove(utility.get_boolean_model(sport=bop.activity_name))

        cm, bm = bop.rebuild_models()
        self.assertTrue(Path.exists(utility.get_categorical_model(sport=bop.activity_name)))
        self.assertTrue(Path.exists(utility.get_boolean_model(sport=bop.activity_name)))
        #  Now give me a prediction score


    def test_generate_predictions(self):
        rop = RunningOutfitPredictor()
        df_full = pd.DataFrame()
        for i in range(1, 100):
            duration = max(20, round(random.normalvariate(45, 45)))
            distance = round((duration / random.triangular(8, 15, 10.5)), 2)
            light_condition = random.choice([True, False])
            fct = Weather.random_forecast()
            fct.is_daylight = light_condition
            conditions = dict(feels_like=fct.feels_like, wind_speed=fct.wind_speed, pct_humidity=fct.pct_humidity,
                              duration=duration, is_light=light_condition, activity_date=fct.timestamp)
            df = pd.DataFrame(vars(fct))
            predicted_outfit = rop.predict_outfit(**conditions)
            df_outfit = pd.DataFrame.from_records([predicted_outfit])
            df2 = pd.concat([df, df_outfit], axis=1)
            df_full = pd.concat([df_full, df2])
        df_full.to_csv(temp_file_path(f'sample.{random.randrange(0, 10000)}.csv'), index=False)

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
                              'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
                              'heavy_socks': True, 'arm_warmers': False, 'face_cover': False})

    def test_predict_outfit_mild_light(self):
        """
        Testing the prediction for running outfits when there is still sunlight.
        :return:
        """
        rop = RunningOutfitPredictor()
        self.assertDictEqual(
            rop.predict_outfit(**{FctKeys.FEEL_TEMP: 52, FctKeys.WIND_SPEED: 0, FctKeys.HUMIDITY: .82,
                                  Features.DURATION: 50, Features.LIGHT: False}),
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

    def test__score_models(self):
        rop = RunningOutfitPredictor()
        # Need to load up a dataset with known values
        rop.rebuild_models(include_xl=False)
        cat_score, bool_score = rop._score_models()


