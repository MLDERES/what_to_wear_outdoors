import datetime
import logging
import os
import pickle
import random
import warnings
from collections import ChainMap
from pathlib import Path
from typing import List, Dict, ClassVar, Callable, Any, Union
import pandas as pd
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from abc import abstractmethod
import datetime as dt
from what_to_wear_outdoors.weather_observation import FctKeys, WIND_DIRECTION

NOW = dt.datetime.now()

if __package__ == '' or __name__ == '__main__':
    from utility import get_data_path, get_boolean_model, get_categorical_model
else:
    from .utility import get_data_path, get_boolean_model, get_categorical_model

logger = logging.getLogger(__name__)


class OutfitComponent:
    def __init__(self, description, alt_description=None):
        """

        :param description: key which will be used to determine the type
        :param alt_description: if not this item then what else.  (i.e. if not heavy socks then regular socks)
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
        self._categorical_labels = [*self._categorical_targets.keys()]
        self._boolean_targets = ['ears_hat', 'gloves', 'heavy_socks', 'arm_warmers', 'face_cover', ]
        self._outfit_classes = self._categorical_labels + self._boolean_targets
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
    @abstractmethod
    def activity_name(self) -> str:
        return 'default'

    @property
    def outfit_component_options(self):
        opt = self._categorical_targets.copy()
        for i in self._boolean_targets:
            opt[i] = ['n', 'y']
        return opt


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


class RoadbikeOutfitMixin(BaseActivityMixin):
    def __init__(self):
        super(RoadbikeOutfitMixin, self).__init__()
        self._local_outfit_components = {'ears_hat': OutfitComponent('ear covers'),
                                         'Boot': OutfitComponent('insulated cycling boot'),
                                         }

    @property
    def activity_name(self):
        return 'Roadbike'

class Features:
    DURATION = 'duration'
    LIGHT = 'is_light'
    DISTANCE = 'distance'
    DATE = 'activity_date'

class BaseOutfitPredictor(BaseActivityMixin):
    """ This class provides the common methods required to predict the clothing need to go outside for a given
        activity.
    """
    __outfit: Dict[str, str]

        
    all_features = { 'scalar': [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY,
                                FctKeys.PRECIP_PCT, FctKeys.REAL_TEMP, FctKeys.HEAT_IDX,
                                Features.DURATION, Features.DISTANCE, Features.DATE],
                     'categorical': [Features.LIGHT, FctKeys.WIND_DIR, FctKeys.CONDITION]}

    def __init__(self):
        super(BaseOutfitPredictor, self).__init__()
        # TODO: When we get good enough add in a few more categorical variables
        self._supported_features = {'scalar': [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY,
                                               Features.DURATION],
                                    'categorical': [Features.LIGHT]}
        self.__outfit = {}
        self._temp_f = 0
        self._sample_data = None
        self._have_sample_data = False
        logger.debug("Creating an instance of %s", self.__class__.__name__)

    @property
    def prediction_labels(self):
        """
        Column names for items of clothing that are going to be predicted
        :return:
        """
        return self._outfit_classes

    @property
    def every_feature(self) -> List[str]:
        """
        Return every single feature that is available, even if it's not yet supported
        :return:
        """
        names = []
        for v in self.all_features.values():
            for n in v:
                names.append(n)
        return names

    @property
    def features(self) -> List[str]:
        """ All the predictor names (factors that contribute to predicting the clothing options) regardless of type

        :return: list of predictor names
        """
        names = []
        for v in self._supported_features.values():
            for n in v:
                names.append(n)
        return names

    # TODO: Handle Imperial and Metric measures (Celsius and kph wind speed)
    def predict_outfit(self, **kwargs) -> Dict[str, Union[str, bool]]:
        """ Predict the clothing options that would be appropriate for an outdoor activity.
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

    def add_to_sample_data(self, forecast, outfit, athlete_name='default', **kwargs):
        """
        Add a row of sample data to a model file.
        :param forecast:
        :param outfit:
        :param athlete_name:
        :return:
        """
        self._have_sample_data = True
        # Check to see if we have a dataframe already
        # If there is no dataframe, then create one
        if self._sample_data is None:
            self._sample_data = self.get_dataframe_format()

        # Add the required lines to the dataframe from the supplied info
        forecast = vars(forecast) if type(forecast) is not dict else forecast
        fields = {x: outfit[x] for x in self.prediction_labels if x in outfit}
        fields.update({x: forecast[x] for x in self.every_feature if x in forecast})
        fields.update({x: kwargs[x] for x in self.every_feature + self.prediction_labels if x in kwargs})
        fields.update({'Athlete':athlete_name, 'activity': self.activity_name})
        record_number = len(self._sample_data)
        for k, v in fields.items():
            self._sample_data.loc[record_number, k] = v
        return self._sample_data

    def write_sample_data(self, filename=""):
        """
        Write the sample data to the file specified
        :param filename:
        :return: the filename of the where the data was written
        """
        default_file_name = f'outfit_data_{NOW.day}{NOW.month}{NOW.hour}{NOW.minute}.csv'
        fn = get_data_path(default_file_name if filename == "" else filename)
        if not self._have_sample_data:
            ReferenceError('No sample data has been created yet.')
        logger.info(f'Writting sample data to {fn}')
        # Take out the first row that we used to format the dataset
        self._sample_data.to_csv(fn)
        return fn


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
    def outfit_(self) -> dict:
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
        # 'feels_like', 'pct_humidity', 'ears_hat', 'outer_layer',
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
        df = pd.DataFrame()
        if filename == 'all':
            # Read every file in the data directory that has a .csv extension
            p = get_data_path("")
            assert (p.is_dir(), f"get_data_path() doesn't point to a valid directory {p}")

            for f in p.glob('*.csv'):
                df = pd.concat([df, pd.read_csv(f)], ignore_index=True)

    
        elif filename == "what i wore running.xlsx":
            data_file = get_data_path(filename)
            logger.debug(f'Reading data from {data_file}')
            #  Import and clean data
            df = pd.read_excel(data_file, sheet_name='Activity Log2',
                                    skip_blank_lines=True,
                                    true_values=['Yes', 'yes', 'y'], false_values=['No', 'no', 'n'],
                                    dtype={'Time': datetime.time}, usecols='A:Y')
        else:
            # Read the file with the name specified
            data_file = get_data_path(filename)
            logger.debug(f'Reading data from {data_file}')
            assert (data_file.exists(), f"{data_file} doesn't point to a valid file")
            df = pd.read_csv(data_file)

        return self._clean_dataframe(df)

    def _clean_dataframe(self, df):
        df.dropna(how='all', inplace=True)
        df.fillna(value=False, inplace=True)
        df.rename(
            {'Date': 'activity_date', 'Time': 'activity_hour', 'Activity': 'activity', 'Distance': Features.DISTANCE,
             'Length of activity (min)': Features.DURATION, 'Condition': FctKeys.CONDITION, 'Light': Features.LIGHT,
             'Wind': FctKeys.WIND_SPEED, 'Wind Dir': FctKeys.WIND_DIR, 'Temp': FctKeys.REAL_TEMP,
             'Feels like': FctKeys.FEEL_TEMP, 'Humidity': FctKeys.HUMIDITY,
             'Hat-Ears': 'ears_hat', 'Outer Layer': 'outer_layer', 'Base Layer': 'base_layer',
             'Arm Warmers': 'arm_warmers', 'Jacket': 'jacket', 'Gloves': 'gloves',
             'LowerBody': 'lower_body', 'Shoe Covers': 'shoe_cover', 'Face Cover': 'face_cover',
             'Heavy Socks': 'heavy_socks', },
            axis='columns', inplace=True, )
        # Now deal with the special cases
        df.drop(['Feel', 'Notes', 'Hat', 'Ears', 'activity_hour'], axis='columns', inplace=True, errors='ignore')
        # Establish the categorical variables
        df['activity_length'] = pd.cut(df.duration, bins=[0, 31, 61, 121, 720],
                                            labels=['short', 'medium', 'long', 'extra long'])
        df['lower_body'] = \
            df['lower_body'].astype(CategoricalDtype(categories=self._leg_categories, ordered=True))
        df['outer_layer'] = \
            df['outer_layer'].astype(CategoricalDtype(categories=self._layer_categories, ordered=True))
        df['base_layer'] = \
            df['base_layer'].astype(CategoricalDtype(categories=self._layer_categories, ordered=True))
        return df

    def get_dataframe_format(self):
        df = self.prepare_data().copy()
        return df.truncate(after=-1)


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

    @outfit_.getter
    def outfit_(self):
        return self.__outfit


class RunningOutfitPredictor(BaseOutfitPredictor, RunningOutfitMixin):
    def __init__(self, ):
        """"""
        super(RunningOutfitPredictor, self).__init__()


class BaseOutfitTranslator(BaseActivityMixin):
    """ This class provides for dealing with the outfit components to full sentence replies

    """
    _base_statements: ClassVar[Dict[str, str]] = \
        {'opening':
             [f'It looks like',
              f'Oh my,',
              f'Well,'],
         'weather':
             [f'the weather should be',
              f'Weather underground says'],
         'clothing':
             [f'I suggest wearing',
              f'Based on the weather conditions, you should consider',
              f'Looks like today would be a good day to wear',
              f'If I were going out I''d wear'],
         'closing':
             [f'Of course, you should always',
              f'It would be insane not to wear',
              f'Also, you should always wear',
              f'And I never go out without']}

    def __init__(self):
        super(BaseOutfitTranslator, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating an instance of %s", self.__class__.__name__)

        self._temp_offset = 0
        #
        self._condition_map: Dict[str, Callable[[Any, Dict[str, Any]], str]] = \
            {FctKeys.FEEL_TEMP: self._get_temp_phrase,
             FctKeys.WIND_SPEED: self._get_windspeed_phrase,
             FctKeys.CONDITION: self._get_weather_condition_phrase,
             FctKeys.PRECIP_PCT: self._get_precipitation_phrase
             }
        self._local_statements = {}

    def _canned_statements(self, key) -> List[str]:
        """
            Get the collection of statement prefixes for different sentence parts
            :param key: one of 'opening', 'clothing', 'always', 'weather'
            :return: list of f-strings suitable to use to build up the replies
        """
        return ChainMap(self._local_statements, self._base_statements)[key]

    def _opening_statements(self):
        return self._canned_statements('opening')

    def _clothing_statements(self):
        return self._canned_statements('clothing')

    def _closing_statements(self):
        return self._canned_statements('closing')

    def _weather_statements(self):
        return self._canned_statements('weather')

    def _get_condition_phrase(self, condition_type, condition, all_conditions=None):
        """
        For a particular condition, call the associated function to help build out the condition phrase

        For instance, if the condition_type is temp and the value is 40 then it may reply with
        'it's going to be chilly'.  Or if the condition_type is humidity and the value is 90% the reply may be
        that it going to be muggy.

        The base class supports 'temp','wind_speed', and 'condition'
        :param condition_type: a condition specified in the __condition_map keys for this translator
        :param condition: the value of that condition
        :return: a phrase which describes the condition, if there is a function that translates this condition for
        the particular activity.
        """
        if condition_type in self._condition_map:
            cond_func = self._condition_map[condition_type]
            return cond_func(condition, all_conditions)

    @staticmethod
    def _get_precipitation_phrase(pop_chance, all_conditions=None):
        return None

    @staticmethod
    def _get_windspeed_phrase(wind_speed, all_conditions=None):
        """
        Given the wind speed, prepare a description of the wind conditions
        :param wind_speed: the wind speed in MPH
        :return: description of the wind condition
        """
        assert isinstance(float(wind_speed), float)
        reply = ""
        if wind_speed < 10:
            reply = "calm"
        elif wind_speed < 15:
            reply = "breezy"
        elif wind_speed < 20:
            reply = "windy"
        else:
            reply = "very windy"
        d = None
        if all_conditions and FctKeys.WIND_DIR in all_conditions:
            direction = all_conditions[FctKeys.WIND_DIR]
            d = WIND_DIRECTION[direction] if direction in WIND_DIRECTION else None
        reply = f'{reply}, {wind_speed} miles per hour out of the {d}' if d else reply
        return reply

    @staticmethod
    def _get_temp_phrase(temp_f, all_conditions=None):
        """ Given a temperature (Farenheit), return a key (condition) used
            to gather up configuratons
        """
        assert isinstance(float(temp_f), float)
        t = temp_f
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
    def _get_weather_condition_phrase(cond, all_conditions=None):
        return cond

    # These are defined as properties so that they could be overridden in subclasses if desired
    @property
    def opening_prefix(self):
        return random.choice(self._opening_statements()) + " it is going to be "

    @property
    def weather_prefix(self):
        return random.choice(self._opening_statements()) + " it is going to be"

    @property
    def clothing_prefix(self):
        # For example
        # A:  I suggest you wear .....
        return random.choice(self._clothing_statements())

    @property
    def closing_prefix(self):
        # For example
        #  A:  Of course, ALWAYS wear .... (helmet / sunscreen)
        return random.choice(self._closing_statements())

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

    def build_reply(self, outfit_items, conditions=None) -> str:
        """ Once the outfit has been predicted this will return the human readable string response

        :type outfit_items: dictionary[str,bool]
        :param outfit_items: the list of items that should be considered, these need to match the names in the
        outfit_components otherwise we have a problem.
        :type conditions dictionary[str, obj]
        :param conditions: a collection of conditions which help to personalize the recommendation
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
                oi = self._get_component_description(i[0], i[1])
            else:
                oi = self._get_component_description(i[1])
            outfit_items_descriptions.append(oi)

        reply = self._build_opening()
        reply_conditions = ""
        if conditions:
            replies = [self._get_condition_phrase(cond_type, cond) for cond_type, cond in conditions.items()]
            reply_conditions += self._build_generic_from_list(replies)
        reply += f'{reply_conditions}.\n' \
            f'{self._build_reply_main(outfit_items_descriptions)}. {self._build_always_reply()}'
        return reply

    def _build_opening(self):
        return self.opening_prefix

    def _build_always_reply(self):
        reply_always = ""
        return reply_always

    def _build_reply_main(self, outfit_item_descriptions: [str]) -> object:
        if outfit_item_descriptions is None:
            raise (ValueError())
        reply_clothing = f'{self.clothing_prefix} {self._build_generic_from_list(outfit_item_descriptions)}'
        return reply_clothing

    @staticmethod
    def _build_generic_from_list(aspects) -> str:
        ''' Build a reply from the list of aspects (outfit components, weather conditions, etc)

        :param aspects: list of things to be put into a full sentence reply
        :return: full sentence reply given the list of items passed to the function
        '''
        reply = ""
        for item in aspects:
            if item is not None:
                reply += f'{item}, '
        reply = reply.strip(', ')
        pos = reply.rfind(',')
        # Here we are just taking out the last , to replace it with 'and'
        if pos > 0:
            reply = reply[:pos] + " and" + reply[pos + 1:]
            reply = reply.replace("and and", "and")
        return reply


class RunningOutfitTranslator(BaseOutfitTranslator, RunningOutfitMixin):
    def __init__(self, ):
        """"""
        super(RunningOutfitTranslator, self).__init__()


class RoadbikeOutfitTranslator(BaseOutfitTranslator, RoadbikeOutfitMixin):
    def __init__(self):
        super(RoadbikeOutfitTranslator, self).__init__()
