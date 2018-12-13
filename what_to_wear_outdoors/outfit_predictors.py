from collections import ChainMap
from random import random
from typing import Dict, Any

if __package__ == '' or __name__ == '__main__':
    from utility import get_model_name, get_data_path, get_model
else:
    from .utility import get_model_name, get_data_path, get_model

class BaseOutfitTranslator:
    """ This class provides for dealing with the outfit components to full sentence replies

    """
    # _local_outfit_descr should be overriden in sub-classes if there is a desire to modify the descriptions
    # of the outfit_components_descr.  ChainMap first looks in _local_outfit_descr for a key value then looks to the 
    # base class definition.  If a subclass wishes to remove or not consider a component, then set the key value to 
    # None and it will be ignored.
    _local_outfit_descr = dict()
    outfit_components_descr = dict(long_sleeves={'name': 'a long-sleeved shirt'},
                                      short_sleeves={'name': 'a short-sleeved shirt'},
                                      shorts={'name': 'shorts'},
                                      calf_sleeves={'name': 'calf sleeves'},
                                      ears_hat={'name': 'ear covers'},
                                      gloves={'name': 'full fingered gloves'},
                                      jacket={'name': 'a windbreaker'},
                                      sweatshirt={'name': 'a sweatshirt or heavier long-sleeve outwear'},
                                      tights={'name': 'tights'},
                                      heavy_socks={'name': 'wool or insulated socks', 'false_name': 'regular socks'})
    _temp_offset = 0


    def __init__(self):
        pass

    def _get_condition_for_temp(self, temp_f):
        """ Given a temperature (Farenheit), return a key (condition) used
            to gather up configuratons
        """
        t = int(temp_f) + self._temp_offset
        if t < 40:
            condition = "cold"
        elif t < 48:
            condition = "cool"
        elif t < 58:
            condition = "mild"
        elif t < 68:
            condition = "warm"
        elif t < 75:
            condition = "very warm"
        else:
            condition = "hot"
        return condition

    # These are defined as properties so that they could be overridden in subclasses if desired
    @property
    def initial_prefix(self):
        return random.choice(self._response_prefixes["initial"]) + " it is going to be"

    @property
    def clothing_prefix(self):
        # For example
        # A:  I suggest you wear .....
        return random.choice(self._response_prefixes["clothing"])

    @property
    def clothing_descriptions(self) -> ChainMap:
        """ A collection of the clothing items supported by this translator.  Should be a dictionary in the form of
        [item_name]['name'] and [item_name]['false_name'] optional
        for instance {'heavy_socks': {'name':'long or heavy socks', false_name:'regular socks'} ...}

        :return: ChainMap of items keyed by item
        """
        return ChainMap(self._local_outfit_descr, self.outfit_components_descr)

    def clothing_item_keys(self) -> [str]:
        ''' The clothing items we are expecting to see for this class

        :return:
        '''
        return [*self.clothing_descriptions.keys()]


    @property
    def always_prefix(self):
        # For example
        #  A:  Of course, ALWAYS wear .... (helmet / sunscreen)
        return random.choice(self._response_prefixes["always"])

    def _get_component_description(self, item_key, false_name=False) -> str:
        """ Determine the proper description of the items that are to be worn (or not)

        This method allows for subclasses to override the default descriptions or to provide alternative
        'false_names'.  False names is a simple way to handle the situation where there are only two options -
        for instance heavy_socks.  If not recommending heavy socks then we recommend regular socks.
        It would be irregular to suggest wearing two kinds of socks, so two alternatives makes sense

        :param item_key: the item for which the description should be fetched
        :param false_name:  if the algorithm says don't get this value, then if there is an alternative return that
        :return: string with the description of the item component, None if not applicable
        """
        item_description = None
        if (self.clothing_descriptions is not None) and item_key in self.clothing_descriptions:
            item = self.clothing_descriptions[item_key]
        if item is not None:
            if false_name:
                item_description = item['false_name'] if 'false_name' in item else None
            else:
                item_description = item['name'] if 'name' in item else None
        return item_description

    def build_reply(self, outfit_items, feels_like_temp):
        """ Once the outfit has been predicted this will return the human readable string response
        
        :type outfit_items: dictionary[str,bool]
        :param outfit_items: the list of items that should be considered, these need to match the names in the
        outfit_components otherwise we have a problem.
        :type feels_like_temp float
        :param feels_like_temp: the outdoor temperature in degrees F
        :return: string representing the human readable outfit options
        """
        if outfit_items is None:
            return ""
        
        outfit_items_descriptions = [self._get_component_description(name, use_false) 
                                     for name, use_false in outfit_items]
        
        reply_temperature = f'{self.initial_prefix} {self._get_condition_for_temp(feels_like_temp)}: ' \
            f'{self.feels_like_temp} degrees'
        reply = f'{reply_temperature}.\n{self._build_reply_main(outfit_items_descriptions)}. {self._build_always_reply()}'
        return reply

    
    def _build_always_reply(self):
        reply_always = ""
        return reply_always

    def _build_reply_main(self, outfit_items: [str]) -> object:
        if outfit_items is None:
            raise (ValueError())
        reply_clothing = f'{self.clothing_prefix} {self._build_generic_from_list(outfit_items)}'
        return reply_clothing

    @staticmethod
    def _build_generic_from_list(outfit_items) -> str:
        ''' Build a reply from the list of outfit descriptions
        
        :param outfit_items: list of item descriptions to be put into a full sentence reply
        :return: full sentence reply given the list of items passed to the function
        '''
        reply = ""
        for item in outfit_items:
            if item is not None:
                reply += f'{item}, '
        reply = reply.strip(', ')
        pos = reply.rfind(',')
        # Here we are just taking out the last , to replace it with 'and'
        if pos > 0:
            reply = reply[:pos] + " and" + reply[pos + 1:]
            reply = reply.replace("and and", "and")
        return reply

class BaseOutfitPredictor():
    outfit_components = ['head', 'base_layer', 'outer_layer', 'lower_body', 'socks', 'gloves']
    _supported_predictors = ['feels_like', 'wind_speed', 'humidity', 'is_light']

    def __init__(self):
        _outfit = {}
        _temp_f = 0

    # TODO: Handle Imperial and Metric measures (Celcius and kph windspeed)
    def predict_outfit(self, duration, verbose=False, **kwargs: object, ):
        """ Predict the clothing options that would be appropriate for an outdoor activity.

        :param verbose:  if True then return a string response suitable for human consumption, if False, return
        a dictionary of the components in the outfit
        :type kwargs: additional forecasting features supported by the particular activity class see
        _supported_predictors contains this list of useful arguments
        :param duration: length of activity in minutes
        :return: dictionary of outfit components, keys are defined by output components, if none specified the consider
        all of the items
        """

        ''' Are the two models up to date?' \
        ' If yes, then we can just open the models and do the predicting' \
        ' if not then we are going to have to do the heavy lifting to create new models'
        '''
        utility.get_data('what i wore running.xlsx')
        pass

    def predict_outfit(self, duration, fct, verbose=False):
        """ Predict the clothing options that would be appropriate for an outdoor activity.

        :param verbose: if True then return a string response suitable for human consumption, if False, return
        a dictionary of the components in the outfit
        :param duration: length of activity in minutes
        :param fct:
        :return: dictionary of outfit components, keys are defined by output components
        """
        pass

    @property
    def outfit_(self):
        return self.__outfit


class RunningOutfitTranslator(BaseOutfitTranslator):
    _local_outfit_components: Dict[str, Dict[str, str]] = dict(ears_hat={'name': 'a hat or ear covers'},
                                                               sweatshirt={'name': 'a sweatshirt'})


class BikingOutfitTranslator(BaseOutfitTranslator):
    _local_outfit_components: Dict[str, Dict[str, str]] = dict(ears_hat={'name': 'ear covers'},
                                                               toe_cover={'name': 'toe covers'})

if __name__ == '__main__':
    bot = BaseOutfitTranslator()
    r = bot._build_generic_from_list(['short sleeves', 'socks', 'shoes'])
    print (r)
