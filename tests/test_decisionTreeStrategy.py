import os
import random

import pytest
from pytest import fixture, mark
import pandas as pd
from what_to_wear_outdoors import model_strategies, FctKeys, Features, config, \
    RunningOutfitPredictor, get_training_data_filepath, Weather, get_test_data_filepath, get_data_path
from what_to_wear_outdoors.model_strategies import SingleDecisionTreeStrategy, DualDecisionTreeStrategy


@fixture
def categorical_targets():
    return {'outer_layer': config.layer_categories,
            'base_layer': config.layer_categories,
            'jacket': config.jacket_categories,
            'lower_body': config.leg_categories,
            'shoe_cover': config.shoe_cover_categories,
            }


@fixture
def boolean_labels():
    return ['ears_hat', 'gloves', 'heavy_socks', 'arm_warmers', 'face_cover', ]


@fixture
def label_list(categorical_targets, boolean_labels):
    return list(categorical_targets.keys()) + boolean_labels


@fixture
def scalar_features():
    return [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY,
            FctKeys.PRECIP_PCT, FctKeys.REAL_TEMP,
            Features.DURATION, Features.DISTANCE, Features.DATE]


@fixture
def categorical_features():
    return [Features.LIGHT, FctKeys.WIND_DIR, FctKeys.CONDITION]


@fixture
def features(scalar_features, categorical_features):
    return {'scalar': scalar_features, 'categorical': categorical_features}


@fixture
def feature_list(features):
    names = []
    for v in features.values():
        for n in v:
            names.append(n)
    return names


@fixture
def ddt_strategy(feature_list, categorical_targets, boolean_labels):
    return DualDecisionTreeStrategy('run', feature_list, categorical_targets, boolean_labels)


@fixture
def sdt_strategy(feature_list, label_list):
    return SingleDecisionTreeStrategy('run', feature_list, label_list)


@fixture
def random_forecasts():
    d2 = [vars(Weather.random_forecast()) for _ in range(0, 100)]
    df = pd.DataFrame(d2)
    df[Features.DURATION] = [max(20, round(random.normalvariate(45, 45))) for _ in range(0, 100)]
    df[Features.DISTANCE] = round(df['duration'] / (random.triangular(8, 15, 10.5)), 2)
    df[Features.LIGHT] = [random.choice([True, False]) for _ in range(0, 100)]
    return df


@fixture
def training_data():
    data_file = get_data_path(config.training_data_filename).with_suffix('.csv')
    assert data_file.exists(), f"{data_file} does not point to a valid file"
    return pd.read_csv(data_file)


def test_save_load_model_ddt(ddt_strategy):
    p = ddt_strategy.save_model('default')
    o2 = model_strategies.load_model(p, 'default', 'run')
    assert ddt_strategy.features == o2.features
    assert ddt_strategy.strategy_id == o2.strategy_id
    os.remove(p)


def test_save_load_model_sdt(sdt_strategy):
    p = sdt_strategy.save_model('default')
    o2 = model_strategies.load_model(p, 'default', 'run')
    assert sdt_strategy.features == o2.features
    assert sdt_strategy.strategy_id == o2.strategy_id
    os.remove(p)


def test_dict_of_predictors(feature_list, categorical_targets, boolean_labels, label_list):
    _predictors = {'ddt': DualDecisionTreeStrategy('run', features=feature_list,
                                                   categorical_targets=categorical_targets,
                                                   boolean_labels=boolean_labels),
                   'sdt': SingleDecisionTreeStrategy('run', features=feature_list,
                                                     labels=label_list)}
    assert type(_predictors['ddt']) is DualDecisionTreeStrategy
    assert type(_predictors['sdt']) is SingleDecisionTreeStrategy


@mark.parametrize("model_strategy,expected", [('ddt', DualDecisionTreeStrategy),
                                              pytest.param('sdt', SingleDecisionTreeStrategy, marks=pytest.mark.xfail)])
def test_get_model_by_strategy(model_strategy, expected):
    """

    :type expected: IOutfitPredictorStrategy
    """
    rop = RunningOutfitPredictor()
    i = type(rop.get_model_by_strategy(model_strategy))
    assert i is expected


def test_predict_outfits(random_forecasts):
    rop = RunningOutfitPredictor()
    model = rop.get_model_by_strategy('ddt')
    rf = random_forecasts
    df = model.predict_outfits(rf)
    print(f'{df}')


def test_ddt_fit(ddt_strategy):
    pass  # o = ddt_strategy


def test_something():
    rop = RunningOutfitPredictor()
    tr_file = get_training_data_filepath()
    df = rop.ingest_data(tr_file)


def test__data_fixup(training_data):
    rop = RunningOutfitPredictor()
    df = rop._data_fixup(training_data)
    print(f'{df.head()}')


@mark.parametrize('drop_known', [True, False])
def test__score_models(drop_known):
    rop = RunningOutfitPredictor()
    # Need to load up a dataset with known values
    mdl = rop.get_model_by_strategy('ddt')
    df = rop.ingest_data(get_test_data_filepath())
    col_scores, overall_score = mdl.score(df, drop_known)
    print(f'\n\nScore (Drop 100% = {drop_known}) -- {overall_score}\nColumn Scores:\n{col_scores}')
