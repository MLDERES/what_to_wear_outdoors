import os
import random
from pathlib import Path
from unittest import TestCase
from sklearn.multioutput import MultiOutputClassifier

from what_to_wear_outdoors import utility, Weather, FctKeys
from what_to_wear_outdoors.outfit_predictors import BaseOutfitPredictor, RunningOutfitPredictor


class TestBaseOutfitPredictor(TestCase):

    def test_get_dataframe_format(self):
        rop = RunningOutfitPredictor()
        df = rop.get_dataframe_format()
        pass

    def test_prepare_data(self):
        bop = BaseOutfitPredictor()
        df = bop.prepare_data()
        print(f'{df.columns}')

    def test_predictors(self):
        bop = BaseOutfitPredictor()
        p = bop.features
        self.assertListEqual([FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY, 'duration', 'is_light'], p)

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
            rop.predict_outfit(**{'feels_like': 50, 'wind_speed': 5, 'pct_humidity': .80, 'duration': 30,
                                  'is_light': True}),
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
            rop.predict_outfit(**{'feels_like': 52, 'wind_speed': 0, 'pct_humidity': .82, 'duration': 50,
                                  'is_light': False}),
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
            rop.predict_outfit(**{'feels_like': 61, 'wind_speed': 0, 'pct_humidity': .42, 'duration': 31,
                                  'is_light': True}),
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
            rop.predict_outfit(**{'feels_like': 72, 'wind_speed': 9, 'pct_humidity': .38, 'duration': 55,
                                  'is_light': True}),
            {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
             'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
             'heavy_socks': False, 'arm_warmers': False, 'face_cover': False})

    def test_add_to_sample_data(self):
        rop = RunningOutfitPredictor()
        f = Weather.random_forecast()
        outfit = {'outer_layer': 'Short-sleeve', 'base_layer': 'Short-sleeve', 'jacket': 'None',
             'lower_body': 'Shorts', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
             'heavy_socks': False, 'arm_warmers': False, 'face_cover': False}
        d = min(20,random.normalvariate(45, 45))
        df = rop.add_to_sample_data(f, outfit, duration=d)
        print(df)
        df = rop.add_to_sample_data(vars(f), outfit, duration=d)
        print(df)

