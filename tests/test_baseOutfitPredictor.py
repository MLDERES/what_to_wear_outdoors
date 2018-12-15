import os
from pathlib import Path
from unittest import TestCase
from sklearn.multioutput import MultiOutputClassifier

from what_to_wear_outdoors import utility
from what_to_wear_outdoors.outfit_predictors import BaseOutfitPredictor, RunningOutfitPredictor


class TestBaseOutfitPredictor(TestCase):
    def test_prepare_data(self):
        bop = BaseOutfitPredictor()
        df = bop.prepare_data()
        print(f'{df.columns}')

    def test_predictors(self):
        bop = BaseOutfitPredictor()
        p = bop.features
        self.assertListEqual(['feels_like', 'wind_speed', 'pct_humidity', 'is_light'], p)

    def test_rebuild_models(self):
        bop = RunningOutfitPredictor()
        if Path.exists(utility.get_categorical_model(sport=bop.activity_name)):
            os.remove(utility.get_categorical_model(sport=bop.activity_name))
        if Path.exists(utility.get_boolean_model(sport=bop.activity_name)):
            os.remove(utility.get_boolean_model(sport=bop.activity_name))

        cm, bm = bop.rebuild_models()
        self.assertTrue(Path.exists(utility.get_categorical_model(sport=bop.activity_name)))
        self.assertTrue(Path.exists(utility.get_boolean_model(sport=bop.activity_name)))

    def test_predict_outfit(self):
        pred_X = [[30, 10, 25, .8, True], [40, 10, 60, .8, True]]
        #print(f'Outer Base Jacket Lower {mt_forest.predict(pred_X)}')
        # ['duration', 'wind_speed', 'feels_like_temp', 'pct_humidity', 'is_light']
        # pred_X = [[30, 10, 25, .8, True], [40, 10, 60, .8, True]]
        #print(f'Ears, Gloves, Heavy Socks {mt_forest.predict(pred_X)}')

