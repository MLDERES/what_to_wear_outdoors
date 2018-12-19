import os
import random
from pathlib import Path
from pprint import pprint
from unittest import TestCase

from what_to_wear_outdoors import utility, Weather, FctKeys
from what_to_wear_outdoors.outfit_predictors import BaseOutfitPredictor, RunningOutfitPredictor, Features


class TestBaseOutfitPredictor(TestCase):

    def test_get_dataframe_format(self):
        rop = RunningOutfitPredictor()
        df = rop.get_dataframe_format()

    def test_prepare_data_from_xls(self):
        rop = RunningOutfitPredictor()
        df = rop.prepare_data()
        print(f'{df.columns}')

    def test_prepare_data_from_one_csv(self):
        rop = RunningOutfitPredictor()
        df = rop.prepare_data('outfit_data_18122036.csv')
        print(f'{df.columns}')

    def test_prepare_data_from_all_csv(self):
        rop = RunningOutfitPredictor()
        df = rop.prepare_data('all')
        print(f'{df.columns}')

    def test_predictors(self):
        bop = BaseOutfitPredictor()
        p = bop.features
        self.assertListEqual(
            [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY, Features.DURATION, Features.LIGHT], p)

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

    def test_predict_outfit_mild_no_light(self):
        """
        Testing the prediction for running outfits when it's not light out.
        :return:
        """
        rop = RunningOutfitPredictor()
        self.assertDictEqual(
            rop.predict_outfit(
                **{FctKeys.FEEL_TEMP: 50, FctKeys.WIND_SPEED: 5, FctKeys.HUMIDITY: .80, Features.DURATION: 30,
                   Features.LIGHT: True}),
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
        d = max(20, random.normalvariate(45, 45))
        df = rop.add_to_sample_data(f, outfit=outfit, athlete_name='Jim', duration=d)
        pprint(df.to_string())
        df = rop.add_to_sample_data(vars(f), outfit=outfit, athlete_name='Default', duration=d)
        pprint(df.to_string())
