import os
from unittest import TestCase

import pytest
from pytest import fixture, mark

from what_to_wear_outdoors import model_strategies, RunningOutfitMixin, FctKeys, Features, config, \
    RunningOutfitPredictor, get_training_data_filename
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


def test_ddt_fit(ddt_strategy):
    pass  # o = ddt_strategy


def test_something():
    rop = RunningOutfitPredictor()
    tr_file = get_training_data_filename()
    df = rop.ingest_data(tr_file, include_xl=False)


def test__score_models():
    rop = RunningOutfitPredictor()
    # Need to load up a dataset with known values
    mdl = rop.get_model_by_strategy('ddt')
    df = rop.ingest_data('what I wore running.xlsx', False)
    cat_score, bool_score = mdl.score(df)
    print(f'Category score: {cat_score}\nBoolean Score: {bool_score}\n')
