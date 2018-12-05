import logging
import json
import random
import numpy as np
import pickle
from utility import get_model_filename

from weather_observation import Forecast

# TODO: Build response string from ML results
OUTFIT_COMPONENTS = {'calf_sleeves': {'name': 'calf sleeves'},
                     'ears_hat': {'name': 'ear covers'},
                     'gloves': {'name': 'full fingered gloves'},
                     'heavy_socks': {'name': 'wool or insulated socks', 'false_name': 'regular socks'},
                     'jacket': {'name': 'a windbreaker'},
                     'long_sleeves': {'name': 'a long-sleeved shirt'},
                     'short_sleeves': {'name': 'a short-sleeved shirt'},
                     'shorts': {'name': 'shorts'},
                     'sweatshirt': {'name': 'a sweatshirt or heavier long-sleeve outwear'},
                     'tights': {'name': 'tights'}
                     }
# Get the list of keys to the clothing dictionary as a list for convenience
CLOTHING_KEYS = [*OUTFIT_COMPONENTS.keys()]
class BaseOutfit(object):
    """ Base class for different activity options.
        Working off the machine determined models
    """
    ACTIVITY_TYPE = ""  # Should be overridden in child classes
    # Override in other subclasses if this list should be different
    ALWAYS_KEY = "always"
    outfit_component_description = {}
    """ This should be overridden in classes where the default description is not adequate
    """

    def _get_component_description(self, item_key, false_name=False) -> str:
        """ Determine the proper description of the items that are to be worn (or not)
        
        This method allows for subclasses to override the default descriptions or to provide alternative
        'false_names'.  False names is a simple way to handle the situation where there are only two options - 
        for instance heavy_socks.  If not recommending heavy socks then we recommend regular socks.
        It would be irregular to suggest wearing two kinds of socks, so two alternatives makes sense
        
        :param item_key: the item for which the description should be fetched 
        :param false_name:  if the algorithm says don't get this value, then if there is an alternative return that
        :return: string with the description of the item component
        """
        item_dict = OUTFIT_COMPONENTS[item_key]
        if (self.outfit_component_description is not None) and item_key in self.outfit_component_description:
            item_dict = self.outfit_component_description[item_key]
        if false_name:
            item_description = item_dict['false_name'] if 'false_name' in item_dict else None
        else:
            item_description = item_dict['name']
        return item_description

    #  Just a couple of attempts to see what it turns up
    def pred_clothing(self, duration, wind_speed, feel, hum, light=True, item='all'):
        outfit = []
        if item == 'all':
            item_list = CLOTHING_KEYS
        elif type(item) is dict or type(item) is list:
            item_list = item
        else:
            item_list = [item]

        for i in item_list:
            model_file = open(get_model_filename(i, sport=self.ACTIVITY_TYPE), 'rb')
            model = pickle.load(model_file)
            model_file.close()
            pms = np.array([duration, wind_speed, feel, hum, not light, light]).reshape(1, -1)
            prediction = model.predict(pms)
            logging.debug(
                f'{item}: {prediction} Feel:{feel} Humidity:{hum} WS: {wind_speed} Duration:{duration} Light:{light}')
            # TODO: Deal with the case where there are multiple levels of choice
            outfit.append(self._get_component_description(i, not prediction))

        return outfit

    _response_prefixes = {
        "initial":
            ["It looks like",
             "Oh my",
             "Well",
             "Temperature seems",
             "Weather underground says"],
        "clothing":
            ["I suggest wearing",
             "Based on the weather conditions, you should consider",
             "Looks like today would be a good day to wear",
             "If I were going out I'd wear"],
        ALWAYS_KEY:
            ["Of course, you should always",
             "It would be insane not to wear",
             "Also, you should always wear",
             "And I never go out without"]}

    def __init__(self, temp_offset=0):
        self._temp_offset = temp_offset
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating an instance of %s", self.__class__.__name__)

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

    @staticmethod
    def _build_generic_from_list(outfit_items):
        reply = ""
        for outfit_item in outfit_items:
            if outfit_item is not None:
                reply += f'{outfit_item}, '
        reply = reply.strip(', ')
        pos = reply.rfind(',')
        # Here we are just taking out the last , to replace it with 'and'
        if pos > 0:
            reply = reply[:pos] + " and" + reply[pos + 1:]
            reply = reply.replace("and and", "and")
        return reply

    def _build_always_reply(self):
        reply_always = ""
        # if self._always is not None:
        #     reply_always += self.always_prefix
        #     reply_always += self._build_generic_from_dict(self._always)
        return reply_always

    def _build_reply_main(self, outfit_items):
        if outfit_items is None:
            raise (ValueError())
        reply_clothing = f'{self.clothing_prefix} {self._build_generic_from_list(outfit_items)}'
        return reply_clothing

    def build_reply(self, forecast, duration=0):
        """ Response to 'What to Wear' for a given forecast

        :param forecast: a forecast object with at least the following fields provided
        (duration, wind_speed, feels like temperature, and humidity)
        :param duration: length of time the activity will be going on
        :return: a human readable string with options of what to wear
        """
        items = self.pred_clothing(duration=duration, wind_speed=forecast.wind_speed, feel=forecast.feels_like_f,
                                   hum=forecast.humidity, light=forecast.is_daylight, item='all')
        temp_f = forecast.feels_like_f
        reply_temperature = f'{self.initial_prefix} {self._get_condition_for_temp(temp_f)}: ' \
            f'{temp_f} degrees'
        if items is not None:
            reply = f'{reply_temperature}.\n{self._build_reply_main(items)}. {self._build_always_reply()}'
        else:
            reply = reply_temperature
            reply += "Unfortunately, I don't know how to tell you to dress for that condition."
        return reply

    # These are defined as properties so that they could be overridden in subclasses if desired
    @property
    def initial_prefix(self):
        return random.choice(self._response_prefixes["initial"]) + "it is going to be "

    @property
    def clothing_prefix(self):
        # For example
        # A:  I suggest you wear .....
        return random.choice(self._response_prefixes["clothing"])

    @property
    def always_prefix(self):
        # For example
        #  A:  Of course, ALWAYS wear .... (helmet / sunscreen)
        return random.choice(self._response_prefixes["always"])


################################################
# Subclassing clothing option for "Running"
#
################################################
class Running(BaseOutfit):
    ACTIVITY_TYPE = "run"
    outfit_component_description = {'ears_hat': {'name': 'a hat or ear covers'},
                                    'sweatshirt': {'name': 'a sweatshirt'}, }


if __name__ == '__main__':
    r = Running()
    print(r.pred_clothing(duration=30, wind_speed=15, feel=75, hum=0, item=OUTFIT_COMPONENTS, light=True))
    print(r.pred_clothing(duration=30, wind_speed=15, feel=75, hum=0, item='all', light=True))
    print(r.pred_clothing(duration=60, wind_speed=15, feel=35, hum=65, item='all', light=True))
    pred = r.pred_clothing(duration=60, wind_speed=15, feel=35, hum=65, item='all', light=True)
    print(f'{r._build_generic_from_list(pred)}')
