from unittest import TestCase

import pytest
from nose.tools import ok_

from what_to_wear_outdoors import FctKeys
from what_to_wear_outdoors.outfit_predictors import RunningOutfitTranslator, OutfitComponent, \
    RunningOutfitPredictor, BaseOutfitTranslator


class TestOutfitTranslator(BaseOutfitTranslator):
    def __init__(self):
        super(TestOutfitTranslator, self).__init__()
        self._local_outfit_descr = {'Short-sleeve': OutfitComponent('singlet'),
                                    'gloves': None,
                                    'camel_back': OutfitComponent('running camelback'),
                                    'wind_jacket': OutfitComponent('wind jacket', 'rain jacket'),
                                    }

class TestBaseOutfitTranslator(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_build_reply_from_fixed_dictionary(self):
        """  Testing build reply using a defined dictionary of values
        :return:
        """
        rot = RunningOutfitTranslator()
        outfit = {'outer_layer': 'Long-sleeve', 'base_layer': 'None', 'jacket': 'None',
                  'lower_body': 'Shorts-calf cover', 'shoe_cover': 'None', 'ears_hat': False, 'gloves': False,
                  'heavy_socks': False, 'arm_warmers': False, 'face_cover': False}
        print(f'{rot.build_reply(outfit, {"feels_like": 50})}')

    def test_build_reply_from_running_predictor(self):
        """ Build a reply from the running outfit predictor, with a known condition
        feels_like:52 , windspeed 0, pct_humidity: 0.82, duration:50
        :return:
        """
        rop = RunningOutfitPredictor()
        rot = RunningOutfitTranslator()
        rop.predict_outfit(**{FctKeys.FEEL_TEMP: 52, FctKeys.WIND_SPEED: 0,
                              FctKeys.HUMIDITY: .82, 'duration': 50,
                              'is_light': False}),
        print(f'{rot.build_reply(rop.outfit_, {"feels_like": 52})}')

    def test_clothing_description(self):
        bot = BaseOutfitTranslator()
        print(f'{bot.clothing_items}')

    def test__build_generic_from_list_simple(self):
        bot = BaseOutfitTranslator()
        r = bot._build_generic_from_list(['short sleeves', 'socks', 'shoes'])
        self.assertEqual('short sleeves, socks and shoes', r)


    def test__get_component_description_subclass_simple_override(self):
        """
        Test of the get_component_description with subclass to ensure that it gets the child description
        :return:
        """
        tot = TestOutfitTranslator()
        r = tot._get_component_description('Short-sleeve')
        self.assertEqual('singlet', r)

    def test__get_component_description_subclass_new_item(self):
        """
        Test get_component_description w/subclass where child only defines the item
        :return:
        """
        tot = TestOutfitTranslator()
        # This tests to see if we can add an item in the subclass
        r = tot._get_component_description('camel_back')
        self.assertEqual('running camelback', r)

    def test__get_component_description_subclass_delete_item(self):
        """
        Test get_component_description w/subclass where child undefines the parent item
        :return:
        """
        bot = BaseOutfitTranslator()
        tot = TestOutfitTranslator()
        # Testing to ensure that if we set a value to None we get None even though it is defined in super class
        r = bot._get_component_description('gloves')
        self.assertEqual('full fingered gloves', r, 'Failed to get the proper parent description')
        r = tot._get_component_description('gloves')
        self.assertIsNone(r)

    def test__get_component_description_subclass_override_alt_name(self):
        """
        Test get_component_description w/subclass  override for a alternative name
        :return:
        """
        tot = TestOutfitTranslator()
        r = tot._get_component_description('wind_jacket', use_alt_name=True)
        self.assertEqual('rain jacket', r)

    def test__get_component_description_running_override_simple(self):
        """
        Test get_component_description w/subclass  override for a alternative name
        :return:
        """
        run_ot = RunningOutfitTranslator()
        r = run_ot._get_component_description('Short-sleeve')
        self.assertEqual('singlet', r)

@pytest.mark.parametrize("translator,condition,value,extras,expected",
                         [
                             (BaseOutfitTranslator(), FctKeys.FEEL_TEMP, 40, None, 'cool'),
                             (RunningOutfitTranslator(), FctKeys.FEEL_TEMP, 40, None, 'cool'),
                             (RunningOutfitTranslator(), FctKeys.WIND_SPEED, 40, None, 'very windy'),
                             (BaseOutfitTranslator(), FctKeys.WIND_SPEED, 40, None, 'very windy'),
                             (BaseOutfitTranslator(), FctKeys.WIND_SPEED, 40, {FctKeys.WIND_DIR: 'N'},
                              'very windy, 40 miles per hour out of the North'),
                             (RunningOutfitTranslator(), FctKeys.WIND_SPEED, 40, {FctKeys.WIND_DIR: 'N'},
                              'very windy, 40 miles per hour out of the North'),
                         ])
def test__get_condition_phrase(translator, condition, value, extras, expected):
    tx = translator
    reply = tx._get_condition_phrase(condition, value, extras)
    assert reply == expected

@pytest.mark.parametrize("translator", [BaseOutfitTranslator(), RunningOutfitTranslator()])
@pytest.mark.parametrize("items",
                         [
                             ['short sleeves', 'socks', 'shoes', None],
                             [None, 'short sleeves', 'socks', 'shoes'],
                             ['short sleeves', None, 'socks', 'shoes'],
                         ])
def test__get_condition_phrase(translator, items):
    tx = translator
    assert tx._build_generic_from_list(items) == 'short sleeves, socks and shoes'

@pytest.mark.parametrize("translator", [BaseOutfitTranslator(), RunningOutfitTranslator(), TestOutfitTranslator()])
@pytest.mark.parametrize("item,expected,alt_name",
                         [('Long-sleeve', 'a long-sleeved top', False),
                          ('heavy_socks', 'wool or insulated socks', False),
                          ('heavy_socks', 'regular socks', True),
                          ]
                         )
def test__get_component_description(translator, item, expected, alt_name):
    tx = translator
    assert tx._get_component_description(item, use_alt_name=alt_name) == expected
