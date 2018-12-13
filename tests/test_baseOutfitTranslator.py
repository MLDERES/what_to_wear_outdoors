from unittest import TestCase

from what_to_wear_outdoors.outfit_predictors import BaseOutfitTranslator


class TestOutfitTranslator(BaseOutfitTranslator):
    _local_outfit_descr = dict(short_sleeves={'name': 'singlet'},
                               gloves=None,
                               camel_back={'name': 'running camelback'},
                               wind_jacket={'name': 'wind jacket', 'false_name':'rain jacket'}
                               )

class TestBaseOutfitTranslator(TestCase):

    def test__build_generic_from_list(self):
        bot = BaseOutfitTranslator()
        r = bot._build_generic_from_list(['short sleeves', 'socks', 'shoes'])
        self.assertEqual('short sleeves, socks and shoes', r)

        r = bot._build_generic_from_list(['short sleeves', 'socks', 'shoes', None])
        self.assertEqual('short sleeves, socks and shoes', r)

        r = bot._build_generic_from_list([None, 'short sleeves', 'socks', 'shoes'])
        self.assertEqual('short sleeves, socks and shoes', r)

        r = bot._build_generic_from_list(['short sleeves', None, 'socks', 'shoes'])
        self.assertEqual('short sleeves, socks and shoes', r)

    def test__get_component_description(self):
        bot = BaseOutfitTranslator()
        # Test first the base case - ensuring that we are getting the value of the class
        r = bot._get_component_description('long_sleeves')
        self.assertEqual('a long-sleeved shirt', r)
        # Test the false_name option
        r = bot._get_component_description('heavy_socks', false_name=True)
        self.assertEqual('regular socks', r)

        # Initial test of the subclass to ensure that it gets the parent description
        tot = TestOutfitTranslator()
        r = tot._get_component_description('long_sleeves')
        self.assertEqual('a long-sleeved shirt', r)

        r = tot._get_component_description('short_sleeves')
        self.assertEqual('singlet', r)

        # This tests to see if we can add an item in the subclass
        r = tot._get_component_description('camel_back')
        self.assertEqual('running camelback', r)

        # Testing to ensure that if we set a value to None we get None even though it is defined in super class
        r = bot._get_component_description('gloves')
        self.assertEqual('full fingered gloves', r)
        r = tot._get_component_description('gloves')
        self.assertIsNone(r)

        # Test the override for a false name
        r = tot._get_component_description('wind_jacket', false_name=True)
        self.assertEqual('rain jacket', r)

        # Just ensure we get back a list of component names
        r = bot.clothing_item_keys()



