from unittest import TestCase

from what_to_wear_outdoors.outfit_predictors import BaseOutfitTranslator, RunningOutfitTranslator, OutfitComponent


class TestOutfitTranslator(BaseOutfitTranslator):
    def __init__(self):
        super(TestOutfitTranslator, self).__init__()
        self._local_outfit_descr = {'Short - sleeve':OutfitComponent('singlet'),
                                    'gloves':None,
                                    'camel_back':OutfitComponent('running camelback'),
                                    'wind_jacket':OutfitComponent('wind jacket', 'rain jacket'),
                                    }

class TestBaseOutfitTranslator(TestCase):

    def test_clothing_description(self):
        bot = BaseOutfitTranslator()
        print(f'{bot.clothing_items}')
        pass

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
        r = bot._get_component_description('Long - sleeve')
        self.assertEqual('a long-sleeved top', r)
        # Test the false_name option
        r = bot._get_component_description('heavy_socks', use_alt_name=True)
        self.assertEqual('regular socks', r)

        # Initial test of the subclass to ensure that it gets the parent description
        tot = TestOutfitTranslator()
        r = tot._get_component_description('Long - sleeve')
        self.assertEqual('a long-sleeved top', r)

        r = tot._get_component_description('Short - sleeve')
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
        r = tot._get_component_description('wind_jacket', use_alt_name=True)
        self.assertEqual('rain jacket', r)

        run_ot = RunningOutfitTranslator()
        r = run_ot._get_component_description('Short - sleeve')
        self.assertEqual('singlet', r)

