import logging
import json
import random
import numpy as np
import pickle
from utility import get_model_filename

from weather_observation import Forecast

OUTFIT_COMPONENTS = {'calf_sleeves': {'name': 'Calf Sleeves'},
                     'ears_hat': {'name': 'Hat or Ear Cover'},
                     'gloves': {'name': 'Full fingered gloves'},
                     'heavy_socks': {'name': 'Wool or insulated socks', 'false_name': 'Regular socks'},
                     'jacket': {'name': 'Windbreaker'},
                     'long_sleeves': {'name': 'Long-sleeved shirt'},
                     'short_sleeves': {'name': 'Short-sleeved shirt'},
                     'shorts': {'name': 'Shorts'},
                     'sweatshirt': {'name': 'Sweatshirt/heavier long-sleeves'},
                     'tights': {'name': 'Tights'}
                     }
class BaseOutfit(object):
    """ Base class for different activity options.
        Working off the machine determined models
    """
    ACTIVITY_TYPE = ""  # Should be overridden in child classes to know where to look in the JSON config
    # Override in other subclasses if this list should be different
    ALWAYS_KEY = "always"

    _configuration = None

    #  Just a couple of attempts to see what it turns up
    def pred_clothing(self, duration, wind_speed, feel, hum, item='shorts', light=True):
        model_file = open(get_model_filename(item, sport=self.ACTIVITY_TYPE), 'rb')
        model = pickle.load(model_file)
        model_file.close()
        pms = np.array([duration, wind_speed, feel, hum, not light, light]).reshape(1, -1)
        prediction = model.predict(pms)
        df = model.decision_function(pms)
        print(f'{item}: {prediction} Feel:{feel} Humidity:{hum} WS: {wind_speed} Duration:{duration} Light:{light}')
        return prediction, df

    _response_prefixes = {
        "initial":
            ["It looks like ",
             "Oh my ",
             "Well ",
             "Temperature seems ",
             "Weather underground says "],
        "clothing":
            ["I suggest wearing ",
             "Based on the weather conditions, you should consider ",
             "Looks like today would be a good day to wear ",
             "If I were going out I'd wear "],
        ALWAYS_KEY:
            ["Of course, you should always ",
             "It would be insane not to wear ",
             "Also, you should always wear ",
             "And I never go out without "]}

    def __init__(self, temp_offset=0):
        self._temp_offset = temp_offset
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating an instance of %s", self.__class__.__name__)

    def _get_condition_for_temp(self, temp_f):
        """ Given a temperature (Farenheit), return a key (condition) used
            to gather up configuratons
        """
        condition = "cold"
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

    def _load_clothing_options(self):
        """ This is the default option for loading up the clothing options.  It can be overridden in child
            classes if there is other set-up/defaults that child
        """
        with open('../data/clothing-options.json') as file_stream:
            config = json.load(file_stream)
        if config is None:
            logging.error("Unable to load the config file clothing-options.json")
        if self.ACTIVITY_TYPE_KEY == "":
            raise (NotImplementedError())
        self._configuration = config[self.ACTIVITY_TYPE_KEY]

    def _get_outfit(self, temp_f, conditions=None):
        """ This function will return the outfit to suggest, given the temperature and any weather conditions
            that might be important
        """
        if self._configuration is None:
            self._load_clothing_options()

        # Now let's get the outfit based on the temperature
        self._condition_temp = self._get_condition_for_temp(temp_f)

        if self.ALWAYS_KEY in self._configuration:
            self._always = self._configuration[self.ALWAYS_KEY]

        if self._condition_temp in self._configuration:
            self._outfit = self._configuration[self._condition_temp]

    def _build_generic_from_dict(self, dct, keys=None):
        reply = ""
        following_list = False
        if keys is None:
            keys = dct.keys()
        for k in keys:
            if k in dct:
                # Deal with lists
                reply += " and also " if following_list else ""
                if type(dct[k]) is list:
                    if len(reply) > 0:
                        reply += 'and '
                    reply += 'either '
                    reply += ' or '.join(dct[k])
                    following_list = True
                # And non-lists
                elif len(dct[k]) > 0:
                    following_list = False
                    reply += dct[k] + ','
        reply = reply.strip(',')
        pos = reply.rfind(',')
        # Here we are just taking out the last , to replace it with 'and'
        if pos > 0:
            reply = reply[:pos] + " and " + reply[pos + 1:]
            reply = reply.replace("and and", "and")
        return reply

    def _build_always_reply(self):
        reply_always = ""
        if self._always is not None:
            reply_always += self.always_prefix
            reply_always += self._build_generic_from_dict(self._always)
        return reply_always

    def _build_reply_main(self):
        reply_clothing = ""
        if self._outfit is None:
            raise (ValueError())

        reply_clothing += self.clothing_prefix
        reply_clothing += self._build_generic_from_dict(self._outfit, self.BODY_PARTS_KEYS)
        return reply_clothing

    def build_reply(self, forecast):
        # Here's where we are going to build reply
        # A: It looks like it is going to be warm (cold, frigid, chilly, hot, mild, super hot)
        temp = forecast.feels_like_f
        self._get_outfit(temp)
        reply_temperature = self.initial_prefix + self._condition_temp + ". {} degrees.".format(temp)
        if self._outfit is not None:
            reply = reply_temperature + ". " + self._build_reply_main() + ". " + self._build_always_reply()
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


r = Running()
for equip in OUTFIT_COMPONENTS.keys():
    r.pred_clothing(duration=80, wind_speed=5, feel=45, hum=66, item=equip, light=False)
    r.pred_clothing(duration=80, wind_speed=5, feel=45, hum=66, item=equip, light=True)
    r.pred_clothing(duration=30, wind_speed=15, feel=75, hum=0, item=equip, light=True)