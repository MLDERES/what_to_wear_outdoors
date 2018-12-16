import datetime
import logging
import os
import pickle
import warnings
from collections import ChainMap
from pathlib import Path
import random

import pandas as pd
import numpy as np
from numpy.core.multiarray import ndarray
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

if __package__ == '' or __name__ == '__main__':
    from utility import get_data_path, get_boolean_model, get_categorical_model
else:
    from .utility import get_data_path, get_boolean_model, get_categorical_model

logger = logging.getLogger(__name__)


class OutfitComponent:
    def __init__(self, description, alt_description=None):
        """

        :param name: key which will be used to determine the type
        :param alternative_name: if not this item then what else.  (i.e. if not heavy socks then regular socks)
        """
        self._description = description
        self._alt_description = alt_description

    @property
    def description(self):
        return self._description

    @property
    def alt_description(self):
        return self._alt_description


class BaseActivityMixin:
    """ This class abstracts the categories of outfit pieces that are described by the translator and features
    """
    _prediction_labels: ndarray

    # _local_outfit_descr should be overriden in sub-classes if there is a desire to modify the descriptions
    # of the outfit_components_descr.  ChainMap first looks in _local_outfit_descr for a key value then looks to the
    # base class definition.  If a subclass wishes to remove or not consider a component, then set the key value to
    # None and it will be ignored.
    def __init__(self):
        self._local_outfit_descr = dict()
        self.outfit_components_descr = {'Sleeveless': OutfitComponent('a sleeveless top'),
                                        'Short-sleeve': OutfitComponent('a short-sleeved shirt'),
                                        'Long-sleeve': OutfitComponent('a long-sleeved top'),
                                        'Sweatshirt-Heavy': OutfitComponent(
                                            'a sweatshirt or heavier long-sleeve outwear'),
                                        'Rain': OutfitComponent('a rain jacket'),
                                        'Warmth': OutfitComponent('a warm jacket'),
                                        'Wind': OutfitComponent('a windbreaker'),
                                        'Rain-Wind': OutfitComponent('a water repellent windbreaker'),
                                        'Warmth-Rain': OutfitComponent('a warm water repellent jacket'),
                                        'Warmth-Wind': OutfitComponent('an insulated windbreaker'),
                                        'Warmth-Rain-Wind': OutfitComponent(
                                            'an water repellent, insulated windbreaker'),
                                        'Shorts': OutfitComponent('shorts'),
                                        'Shorts-calf cover': OutfitComponent('shorts with long socks or calf sleeves'),
                                        'Capri': OutfitComponent('capri tights'),
                                        'Long tights': OutfitComponent('full length tights'),
                                        'Insulated tights': OutfitComponent('Insulated tights'),
                                        'ears_hat': OutfitComponent('ear covers'),
                                        'gloves': OutfitComponent('full fingered gloves'),
                                        'heavy_socks': OutfitComponent('wool or insulated socks', 'regular socks'),
                                        'arm_warmers': OutfitComponent('arm warmers'),
                                        'face_cover': OutfitComponent('face cover'),
                                        'Toe Cover': OutfitComponent('toe covers'),
                                        }
        self._layer_categories = ['None', 'Sleeveless', 'Short-sleeve', 'Long-sleeve', 'Sweatshirt-Heavy']
        self._jacket_categories = ['None', 'Rain', 'Warmth', 'Wind', 'Rain-Wind',
                                   'Warmth-Rain', 'Warmth-Wind', 'Warmth-Rain-Wind']
        self._leg_categories = ['Shorts', 'Shorts-calf cover', 'Capri', 'Long tights', 'Insulated tights']
        self._shoe_cover_categories = ['None', 'Toe Cover', 'Boots']

        self._categorical_targets = {'outer_layer': self._layer_categories,
                                     'base_layer': self._layer_categories,
                                     'jacket': self._jacket_categories,
                                     'lower_body': self._leg_categories,
                                     'shoe_cover': self._shoe_cover_categories,
                                     }
        self._boolean_targets = \
            self._boolean_labels = ['ears_hat', 'gloves', 'heavy_socks', 'arm_warmers', 'face_cover', ]
        self._outfit_classes = [*self._categorical_targets.keys()] + self._boolean_targets
        self._categorical_labels = [*self._categorical_targets.keys()]
        self._prediction_labels = np.concatenate((self._categorical_labels, self._boolean_labels), axis=None)

        logger.debug("Creating an instance of %s", self.__class__.__name__)

    @property
    def clothing_descriptions(self) -> ChainMap:
        """ A collection of the clothing descriptions for this translator.

        :return: ChainMap of items keyed by item
        """
        return ChainMap(self._local_outfit_descr, self.outfit_components_descr)

    @property
    def clothing_items(self) -> [str]:
        """ The clothing items we are expecting to see for this class

        :return:The clothing items we are expecting to see for this class
        """
        return [*self.clothing_descriptions.keys()]

    @property
    def activity_name(self) -> str:
        return 'default'

    @property
    def prediction_labels(self):
        return self._prediction_labels

class BaseOutfitTranslator(BaseActivityMixin):
    """ This class provides for dealing with the outfit components to full sentence replies

    """
    _response_prefixes = {
        "initial":
            ["It looks like",
             "Oh my,",
             "Well,",
             "Temperature seems",
             "Weather underground says"],
        "clothing":
            ["I suggest wearing",
             "Based on the weather conditions, you should consider",
             "Looks like today would be a good day to wear",
             "If I were going out I'd wear"],
        "always":
            ["Of course, you should always",
             "It would be insane not to wear",
             "Also, you should always wear",
             "And I never go out without"]}

    def __init__(self):
        super(BaseOutfitTranslator, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating an instance of %s", self.__class__.__name__)
        self._temp_offset = 0

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
    def always_prefix(self):
        # For example
        #  A:  Of course, ALWAYS wear .... (helmet / sunscreen)
        return random.choice(self._response_prefixes["always"])

    def _get_component_description(self, item_key, use_alt_name=False) -> str:
        """ Determine the proper description of the items that are to be worn (or not)

        This method allows for subclasses to override the default descriptions or to provide alternative
        'false_names'.  False names is a simple way to handle the situation where there are only two options -
        for instance heavy_socks.  If not recommending heavy socks then we recommend regular socks.
        It would be irregular to suggest wearing two kinds of socks, so two alternatives makes sense

        :param item_key: the item for which the description should be fetched
        :param use_alt_name:  if the algorithm says don't get this value, then if there is an alternative return that
        :return: string with the description of the item component, None if not applicable
        """
        item_description = None
        item = None
        if (self.clothing_descriptions is not None) and item_key in self.clothing_descriptions:
            item = self.clothing_descriptions[item_key]
        if item is not None:
            item_description = item.alt_description if use_alt_name else item.description
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

        # We can iterate this list, but some of it is descriptions, some are true/false components
        # so we need to decide what kind it is and deal with it accordingly

        #  Go through the items we want to output for this translator and just get the descriptions for them
        outfit_items_descriptions = []
        for i in outfit_items.items():
            if i[1] is bool:
                oi = self._get_component_description(i[0],i[1])
            else:
                oi = self._get_component_description(i[1])
            outfit_items_descriptions.append(oi)

        reply_temperature = f'{self.initial_prefix} {self._get_condition_for_temp(feels_like_temp)}: ' \
            f'{feels_like_temp} degrees'
        reply = f'{reply_temperature}.\n' \
            f'{self._build_reply_main(outfit_items_descriptions)}. {self._build_always_reply()}'
        return reply

    def _build_always_reply(self):
        reply_always = ""
        return reply_always

    def _build_reply_main(self, outfit_item_descriptions: [str]) -> object:
        if outfit_item_descriptions is None:
            raise (ValueError())
        reply_clothing = f'{self.clothing_prefix} {self._build_generic_from_list(outfit_item_descriptions)}'
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


class BaseOutfitPredictor(BaseActivityMixin):
    """ This class provides the common methods required to predict the clothing need to go outside for a given
        activity.
    """

    def __init__(self):
        super(BaseOutfitPredictor, self).__init__()
        # TODO: When we get good enough add in a few more categorical variables
        self._supported_features = {'scalar': ['feels_like', 'wind_speed', 'pct_humidity', 'duration'],
                                    'categorical': ['is_light']}
        self._outfit = {}
        self._temp_f = 0
        logger.debug("Creating an instance of %s", self.__class__.__name__)

    @property
    def features(self):
        """ All the predictor names (factors that contribute to predicting the clothing options) regardless of type

        :return: list of predictor names
        """
        names = []
        for v in self._supported_features.values():
            for n in v:
                names.append(n)
        return names

    # TODO: Handle Imperial and Metric measures (Celsius and kph wind speed)
    def predict_outfit(self, verbose=False, **kwargs: object):
        """ Predict the clothing options that would be appropriate for an outdoor activity.

        :param verbose:  if True then return a string response suitable for human consumption, if False, return
        a dictionary of the components in the outfit
        :type kwargs: additional forecasting features supported by the particular activity class see
        _supported_features contains this list of useful arguments
        :param duration: length of activity in minutes
        :return: dictionary of outfit components, keys are defined by output components
        """

        # Now we need to predict using the categorical model and then the boolean model
        #  The categorical model will predict items that can be of more than one type
        #  (i.e. base layer can be sleeveless, short-sleeved or long-sleeve)
        #  boolean is used for categories of 2 or True/False
        #   (i.e. heavy socks, if true then use heavy socks, otherwise we can assume regular socks)
        #   (i.e. arm warmers, if needed True if not then False
        cat_model, bool_model = self.get_models()

        # Load up a dataset with the supplied predictors (from the input parameters)
        # Filter this list based on what is used to predict the outcomes
        # Run the predictor
        # Put the results into a dictionary that can be returned to the caller
        prediction_factors = pd.DataFrame(kwargs, index=[0])
        prediction_factors = prediction_factors[self.features]
        cat_outcomes = cat_model.predict(prediction_factors).reshape(-1)
        bool_outcomes = bool_model.predict(prediction_factors).reshape(-1)

        predictions = np.concatenate((cat_outcomes, bool_outcomes), axis=None)
        results = dict(zip(self.prediction_labels, predictions))

        # Save the last outfit we predicted to a property for easy retrieval
        self.__outfit = results

        return results

    def get_models(self):
        """ Get the categorical and boolean models

        :return: A tuple containing the categorical and boolean models
        """
        ''' Are the two models up to date?' \
            ' If yes, then we can just open the models and do the predicting' \
            ' if not then we are going to have to do the heavy lifting to create new models'
            '''
        raw_data_path = get_data_path('what i wore running.xlsx')
        boolean_model_path = get_boolean_model(sport=self.activity_name)
        cat_model_path = get_categorical_model(sport=self.activity_name)

        if self._are_models_upto_date(cat_model_path, raw_data_path, boolean_model_path):
            # We are good to go with the current models and just need to do the predicting
            (cat_model, bool_model) = self.load_models()
        else:
            # Need to rebuild the models
            (cat_model, bool_model) = self.rebuild_models()
        return cat_model, bool_model

    def _are_models_upto_date(self, cat_model_path, raw_data_path, boolean_model_path):
        """ Determine if the model files exist and if so, is the raw data file newer

        :param cat_model_path:
        :param raw_data_path:
        :param boolean_model_path:
        :return:
        """
        files_exist = Path.exists(cat_model_path) and Path.exists(boolean_model_path)
        bool_is_newer = cat_is_newer = False
        if files_exist:
            bool_is_newer = os.stat(boolean_model_path).st_ctime >= os.stat(raw_data_path).st_ctime
            cat_is_newer = os.stat(cat_model_path).st_ctime >= os.stat(raw_data_path).st_ctime
        return files_exist and bool_is_newer and cat_is_newer

    @property
    def outfit_(self):
        return self.__outfit

    def rebuild_models(self) -> (MultiOutputClassifier, MultiOutputClassifier):
        """  Read the raw data and build the models (one for categorical targets one for boolean targets)
            This function also pickles and saves the models and returns them to the caller in a tuple

        :return: (cat_model, bool_model)
        """
        full_ds = self.prepare_data()

        full_train_ds = full_ds[(full_ds.activity == self.activity_name)].copy()
        # TODO: --SPORT FILTER -- Remove this when we have taken care to handle multiple athletes
        full_train_ds.drop(['Athlete', 'activity', 'activity_date'], axis='columns', inplace=True)

        # 'weather_condition', 'is_light', 'wind_speed', 'wind_dir', 'temp',
        # 'feels_like_temp', 'pct_humidity', 'ears_hat', 'outer_layer',
        # 'base_layer', 'arm_warmers', 'jacket', 'gloves', 'lower_body',
        # 'heavy_socks', 'shoe_cover', 'face_cover', 'activity_month',
        # 'activity_length']
        def drop_others(df, keep):
            drop_cols = [k for k in df.columns if k not in keep]
            df.drop(drop_cols, axis=1, inplace=True)

        # Take out any columns which we haven't specified as features or labels
        drop_others(full_train_ds, self._outfit_classes + self.features)

        # Finally, we are going to be able to establish the model.
        train_X = full_train_ds[self.features]

        cat_targets = list(self._categorical_targets.keys())
        y_class = full_train_ds[cat_targets]
        y_bools = full_train_ds[self._boolean_targets]

        # https://scikit-learn.org/stable/modules/multiclass.html
        # We have to do two models, one for booleans and one for classifiers (which seems dumb)
        # Starting with the booleans
        bool_forest = DecisionTreeClassifier(max_depth=4)
        bool_forest_mo = MultiOutputClassifier(bool_forest)
        bool_forest_mo.fit(train_X, y_bools)
        model_file = get_boolean_model(sport=self.activity_name)
        pickle.dump(bool_forest_mo, open(model_file, 'wb'))

        # Now for the classifiers
        forest = DecisionTreeClassifier(max_depth=4)
        mt_forest = MultiOutputClassifier(forest, n_jobs=-1)
        mt_forest.fit(train_X, y_class)
        model_file = get_categorical_model(sport=self.activity_name)
        pickle.dump(mt_forest, open(model_file, 'wb'))
        return mt_forest, bool_forest_mo

    def prepare_data(self, filename='what i wore running.xlsx'):
        """
        Rebuild both of the models associated with this activity, the categorical and the boolean model
        :return: a DataFrame encapsulating the raw data, cleaned up from the excel file
        """
        data_file = get_data_path(filename)
        logger.debug(f'Reading data from {data_file}')
        #  Import and clean data
        full_ds = pd.read_excel(data_file, sheet_name='Activity Log2',
                                skip_blank_lines=True,
                                true_values=['Yes', 'yes', 'y'], false_values=['No', 'no', 'n'],
                                dtype={'Time': datetime.time}, usecols='A:Y')
        full_ds.dropna(how='all', inplace=True)
        full_ds.fillna(value=False, inplace=True)
        logger.debug(f'Data file shape: {data_file}')
        full_ds.rename(
            {'Date': 'activity_date', 'Time': 'activity_hour', 'Activity': 'activity', 'Distance': 'distance',
             'Length of activity (min)': 'duration', 'Condition': 'weather_condition', 'Light': 'is_light',
             'Wind': 'wind_speed', 'Wind Dir': 'wind_dir', 'Temp': 'temp', 'Feels like': 'feels_like',
             'Humidity': 'pct_humidity',
             'Hat-Ears': 'ears_hat', 'Outer Layer': 'outer_layer', 'Base Layer': 'base_layer',
             'Arm Warmers': 'arm_warmers', 'Jacket': 'jacket', 'Gloves': 'gloves',
             'LowerBody': 'lower_body', 'Shoe Covers': 'shoe_cover', 'Face Cover': 'face_cover',
             'Heavy Socks': 'heavy_socks', },
            axis='columns', inplace=True, )
        # Now deal with the special cases
        full_ds.drop(['Feel', 'Notes', 'Hat', 'Ears', 'activity_hour'], axis='columns', inplace=True, errors='ignore')
        # Establish the categorical variables
        full_ds['activity_month'] = full_ds['activity_date'].dt.month.astype('category')
        full_ds['activity_length'] = pd.cut(full_ds.duration, bins=[0, 31, 61, 121, 720],
                                            labels=['short', 'medium', 'long', 'extra long'])
        full_ds['lower_body'] = \
            full_ds['lower_body'].astype(CategoricalDtype(categories=self._leg_categories, ordered=True))
        full_ds['outer_layer'] = \
            full_ds['outer_layer'].astype(CategoricalDtype(categories=self._layer_categories, ordered=True))
        full_ds['base_layer'] = \
            full_ds['base_layer'].astype(CategoricalDtype(categories=self._layer_categories, ordered=True))

        return full_ds

    def load_models(self):
        """
        Go to the default models directory and load up both the boolean and categorical models
        :return: tuple containing the categorical model and the boolean model
        """
        boolean_model_path = get_boolean_model(self.activity_name)
        cat_model_path = get_categorical_model(self.activity_name)
        with warnings.catch_warnings(), open(cat_model_path, 'rb') as cf, open(boolean_model_path, 'rb') as bf:
            warnings.simplefilter('ignore')
            cat_model = pickle.load(cf)
            bool_model = pickle.load(bf)
        return cat_model, bool_model

    @outfit_.setter
    def outfit_(self, value):
        self._outfit_ = value


class RunningOutfitMixin(BaseActivityMixin):
    """ Provides an override for the components that are unique to running

    """

    def __init__(self):
        super(RunningOutfitMixin, self).__init__()
        self._local_outfit_descr = {'Short-sleeve': OutfitComponent('singlet'),
                                    'Toe cover': None,
                                    'Boot': None,
                                    'face_cover': None,
                                    }
    @property
    def activity_name(self):
        return 'Run'


class RunningOutfitTranslator(BaseOutfitTranslator, RunningOutfitMixin):
    def __init__(self, ):
        """"""
        super(RunningOutfitTranslator, self).__init__()


class RunningOutfitPredictor(BaseOutfitPredictor, RunningOutfitMixin):
    def __init__(self, ):
        """"""
        super(RunningOutfitPredictor, self).__init__()


class RoadbikeOutfitMixin(BaseActivityMixin):
    def __init__(self):
        super(RoadbikeOutfitMixin, self).__init__()
        self._local_outfit_components = {'ears_hat': OutfitComponent('ear covers'),
                                         'Boot': OutfitComponent('insulated cycling boot'),
                                         }

    @property
    def activity_name(self):
        return 'Run'


class RoadbikeOutfitTranslator(BaseOutfitTranslator, RoadbikeOutfitMixin):
    def __init__(self):
        super(RoadbikeOutfitTranslator, self).__init__()
