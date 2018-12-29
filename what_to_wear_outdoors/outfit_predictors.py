import datetime
import logging
import os
import random
from collections import ChainMap
from pathlib import Path
from typing import List, Dict, ClassVar, Callable, Any, Union
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from abc import abstractmethod
import datetime as dt
from what_to_wear_outdoors import config
from what_to_wear_outdoors.utility import get_data_path, get_model_path, get_training_data_path, get_test_data_path

from what_to_wear_outdoors.model_strategies import IOutfitPredictorStrategy, DualDecisionTreeStrategy, \
    SingleDecisionTreeStrategy, load_model
from what_to_wear_outdoors.weather_observation import FctKeys, WIND_DIRECTION

FALSE_VALUES = ['No', 'no', 'n', 'N']
TRUE_VALUES = ['Yes', 'yes', 'y', 'Y']
NOW = dt.datetime.now()
TODAY = dt.datetime.today()


class Features:
    DURATION = 'duration'
    LIGHT = 'is_light'
    DISTANCE = 'distance'
    DATE = 'activity_date'


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
        self._layer_categories = config.layer_categories
        self._jacket_categories = config.jacket_categories
        self._leg_categories = config.leg_categories
        self._shoe_cover_categories = config.shoe_cover_categories
        self._categorical_targets = {'outer_layer': self._layer_categories,
                                     'base_layer': self._layer_categories,
                                     'jacket': self._jacket_categories,
                                     'lower_body': self._leg_categories,
                                     'shoe_cover': self._shoe_cover_categories,
                                     }
        self._categorical_labels = [*self._categorical_targets.keys()]
        self._boolean_labels = ['ears_hat', 'gloves', 'heavy_socks', 'arm_warmers', 'face_cover', ]
        self._outfit_labels = self._categorical_labels + self._boolean_labels
        logger.debug("Creating an instance of %s", self.__class__.__name__)

    @property
    def clothing_descriptions(self) -> ChainMap:
        """ A dictionary for the description of an item based on the activity class.  The keys are available in
        the `clothing_items` property.  This property is used by the translators to provide the appropriate
        description for a particular clothing item which has a generic key.

        For instance, {'Long tights':'full length tights', 'gloves':'full-fingered gloves'}.
        """
        return ChainMap(self._local_outfit_descr, self.outfit_components_descr)

    @property
    def clothing_items(self) -> [str]:
        """ A list of strings that are keys into the `clothing_descriptions` dictionary

        For instance: ['gloves', 'Long tights', 'Capri', 'face_cover']
        """
        return [*self.clothing_descriptions.keys()]

    @property
    @abstractmethod
    def activity_name(self) -> str:
        return 'default'

    @property
    def outfit_component_options(self):
        opt = self._categorical_targets.copy()
        for i in self._boolean_labels:
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

class BaseOutfitPredictor(BaseActivityMixin):
    """ This class provides the common methods required to predict the clothing need to go outside for a given
        activity.
    """
    _outfit_model: IOutfitPredictorStrategy
    __outfit: Dict[str, str]

    all_features = {'scalar': [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY,
                               FctKeys.PRECIP_PCT, FctKeys.REAL_TEMP,
                               Features.DURATION, Features.DISTANCE, Features.DATE],
                    'categorical': [Features.LIGHT, FctKeys.WIND_DIR, FctKeys.CONDITION]}

    def __init__(self):
        super(BaseOutfitPredictor, self).__init__()
        logger.debug("Creating an instance of %s", self.__class__.__name__)

        # TODO: When we get good enough add in a few more categorical variables
        self._supported_features = {'scalar': [FctKeys.FEEL_TEMP, FctKeys.WIND_SPEED, FctKeys.HUMIDITY,
                                               Features.DURATION],
                                    'categorical': [Features.LIGHT]}
        self.__outfit = {}
        self._sample_data = None
        self._have_sample_data = False
        self._active_predictor = None
        self._predictors = {'ddt': DualDecisionTreeStrategy(self.activity_name, features=self.features,
                                                            categorical_targets=self._categorical_targets,
                                                            boolean_labels=self._boolean_labels),
                            'sdt': SingleDecisionTreeStrategy(self.activity_name, features=self.features,
                                                              labels=self.labels)}

    @property
    def labels(self):
        """
        Column names for items of clothing that are going to be predicted
        :return:
        """
        return self._outfit_labels

    @property
    def features(self) -> List[str]:
        """ All the predictor names (factors that contribute to predicting the clothing options)

        :return: list of predictor names
        """
        names = []
        for v in self._supported_features.values():
            for n in v:
                names.append(n)
        return names

    @property
    def outfit_(self) -> dict:
        return self.__outfit

    @outfit_.getter
    def outfit_(self) -> dict:
        return self.__outfit

    # TODO: Handle Imperial and Metric measures (Celsius and kph wind speed)
    def predict_outfit(self, strategy: str = 'ddt', **kwargs) -> Dict[str, Union[str, bool]]:
        """ Predict the clothing options that would be appropriate for an outdoor activity.

        :type strategy: {'ddt','sdt'}, default 'ddt'
        :param strategy: the particular strategy that should be used to make the prediction
        :param kwargs: features supported by this activity class ``_supported_features`` contains
            the list of valid arguments
        :return: dictionary of outfit components, keys are defined by the ``labels_`` property

        >>> rop = RunningOutfitPredictor()...
        >>> rop.predict_outfit(
            **{'feels_like': 69, 'duration': 45, 'temp_f': 69, 'wind_dir': 'E',
            'condition': 'Clear',
            'distance': 4, 'activity': 'Run', 'wind_speed': 15,
            'is_light': True, 'pct_humidity': 20})...

        ['outer_layer': 'Short-sleeve', 'base_layer': None, 'Heavy_socks': False, ...]

        Or more explicitly:

        rop.predict_outfit("feels_like"= 69, "duration" = 45, "temp_f" = 69, "wind_dir" = "E",
                                'condition' = 'Clear', 'distance' = 4, 'activity' = 'Run',
                                'wind_speed' = 15, 'is_light''=True,'pct_humidity'= 20)

        ['outer_layer':'Short-sleeve', 'base_layer':None, 'Heavy_socks':False, ...]

        """

        logger.debug(f'Attempting to get a model for {strategy}')
        mdl = self.get_model_by_strategy(strategy)

        # Load up a dataset with the supplied predictors (from the input parameters)
        # Filter this list based on what is used to predict the outcomes
        # Run the predictor
        # Put the results into a dictionary that can be returned to the caller
        prediction_factors = pd.DataFrame(kwargs, index=[0])
        logger.debug(f'These parameters were passed to the predict_outfit function: {kwargs}')
        predict_X = prediction_factors[mdl.features]
        logger.debug(f'Peeling off the factors from the supplied prediction parameters ({self.features})')

        results = mdl.predict_outfit(predict_X)

        logger.debug(f'All the results for the prediction: {results}')

        # Save the last outfit we predicted to a property for easy retrieval
        self.__outfit = results
        return results

    def get_model_by_strategy(self, model_type: str) -> IOutfitPredictorStrategy:
        """
        Load the proper model specified by the `model_type`, if this model doesn't exist then train this strategy
        from the ..\data\training_data.csv file
        :type model_type: str
        :param model_type: one of 'ddt' - DualDecisionTreeStrategy or 'sdt' - SingleDecisionTreeStrategy others maybe
        made available in the future
        :return: A model that can predict the right outfit based on weather conditions.

        """
        assert model_type in self._predictors
        predictor = self._predictors[model_type]
        model_file = get_model_path(predictor.get_model_filename())
        tr_file = get_training_data_path()
        out_of_date = True

        if Path.exists(model_file) and Path.exists(tr_file):
            train_stamp = os.stat(tr_file).st_mtime
            mdl_stamp = os.stat(model_file).st_mtime
            out_of_date = train_stamp > mdl_stamp

        if out_of_date:
            # Need to train a new model
            df = self.ingest_data(tr_file)
            predictor.fit(df)
            predictor.save_model()
        else:
            # Load up the model we have
            predictor = load_model(model_file)

        self._active_predictor = predictor
        return predictor

    def ingest_data(self, filename='all') -> pd.DataFrame:
        """
        Loads and prepares the data files specified.

        :param filename: specify the file to use to create the model or 'all' if filename extension is .xlsx, then
                         include_xl is ignored
        :return: a DataFrame encapsulating the raw data, cleaned up from the excel file
        """
        df = pd.DataFrame()
        if filename == 'all':
            # Read every file in the data directory that has a .csv extension
            p = get_data_path("")

            for f in p.glob('*.csv'):
                logger.debug(f'Reading data from {f}')
                try:
                    df = pd.concat([df, pd.read_csv(f, true_values=TRUE_VALUES, false_values=FALSE_VALUES)],
                                   ignore_index=True, sort=False)
                except:
                    logger.warning(f'Invalid file format, file = {f}')

        else:
            # Read the file with the name specified
            if get_data_path(filename).suffix == '.xlsx':
                df = pd.concat([df, self._read_xl(filename)])
                include_xl = False
            else:
                data_file = get_data_path(filename).with_suffix('.csv')
                logger.debug(f'Reading data from {data_file}')
                assert data_file.exists(), f"{data_file} does not point to a valid file"
                df = pd.read_csv(data_file)

        return self._data_fixup(df)

    def _data_fixup(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Common cleanup activities once the dataset has been loaded either from the training data in the csv files
        or the test data in the XLSX file.

        :param df: A dataframe containing the features and labels used to either train or predict the model
        :type df: DataFrame
        :return: updated version of the dataframe with only the columns that are relevant for training or predicting

        #todo-list:

        """
        df.fillna(value=False, inplace=True)
        df = self._fix_layers(df)
        df = self._build_data_categories(df)
        if 'activity' in df.columns:
            df = df[(df.activity == self.activity_name)].reset_index(drop=True)
        # TODO:
        df.drop(['Athlete', 'activity', 'activity_date'], axis='columns', inplace=True, errors='ignore')

        # 'weather_condition', 'is_light', 'wind_speed', 'wind_dir', 'temp',
        # 'feels_like', 'pct_humidity', 'ears_hat', 'outer_layer',
        # 'base_layer', 'arm_warmers', 'jacket', 'gloves', 'lower_body',
        # 'heavy_socks', 'shoe_cover', 'face_cover', 'activity_month',
        # 'activity_length']
        def drop_others(df2, keep):
            drop_cols = [k for k in df2.columns if k not in keep]
            df2.drop(drop_cols, axis=1, inplace=True)

        # Take out any columns which we haven't specified as features or labels
        drop_others(df, self._outfit_labels + self.features)
        return df

    def _read_xl(self, filename=get_test_data_path(), sheet_name='Activity Log2') -> pd.DataFrame:
        """
        Create a dataframe from an Excel sheet used to capture actual clothing choices

        :param filename: the Excel file to use (must be in the data directory)
        :param sheet_name: the name of the sheet to read from the Excel file
        :return: a pandas dataframe matching the same format as the csv files that are created from input mode
        """
        data_file = get_data_path(filename)
        logger.debug(f'Reading XLSX data from {data_file}')
        #  Import and clean data
        df_xl = pd.read_excel(data_file, sheet_name=sheet_name,
                              skip_blank_lines=True,
                              true_values=TRUE_VALUES, false_values=FALSE_VALUES,
                              dtype={'Time': datetime.time}, usecols='A:Y')
        df_xl.dropna(how='all', inplace=True)

        df_xl.rename(
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
        df_xl.drop(['Feel', 'Notes', 'Hat', 'Ears', 'activity_hour'], axis='columns', inplace=True, errors='ignore')
        return df_xl

    def _build_data_categories(self, df):
        """
        Establish the proper data types for categorical columns
        :param df:
        :return:
        """
        df['activity_length'] = pd.cut(df.duration, bins=[0, 31, 61, 121, 720],
                                       labels=['short', 'medium', 'long', 'extra long'])
        for col_name, cat_type in [('lower_body', self._leg_categories), ('outer_layer', self._layer_categories),
                                   ('base_layer', self._layer_categories), ('jacket', self._jacket_categories),
                                   ('shoe_cover', self._shoe_cover_categories)]:
            df[col_name] = df[col_name].astype(CategoricalDtype(categories=cat_type, ordered=True))
        return df

    def _fix_layers(self, df) -> pd.DataFrame:
        """
        Ensures that clothing are properly defined.  Simply, if you wearing a jacket then you can only have an outer
        layer if you are also wearing a base layer.  Similarly, if you are not wearing a jacket, then you can only
        have a base layer if you are also wearing an outer layer.


            Jacket + Outer Layer + No Base .. becomes Jacket + base layer + No Outer layer
            Jacket + Outer Layer + Base Layer = good
            No Jacket + Outer Layer + No Base Layer = good
            No Jacket + Base Layer + No Outer Layer = Outer Layer (=Base Layer) + No Base Layer
            No Jacket + Outer Layer + Base Layer = good
        :param df:
        :return:
        """

        no_base = (df['base_layer'] == 'None')
        has_base = ~no_base
        no_outer = (df['outer_layer'] == 'None')
        has_outer = ~no_outer
        no_layer = (no_outer & no_base)
        no_jacket = (df['jacket'] == 'None')
        has_jacket = ~no_jacket

        # Drop all rows where the outer layer and the base layer are both empty
        l = (df.loc[no_layer]).shape
        logger.info(f'dropping {l[0]} rows where no base layer and no outer_layer are identified')
        df = df[~no_layer]

        df3 = df.copy()

        df3.loc[has_jacket & has_outer & no_base, ['base_layer']] = df['outer_layer']
        df3.loc[has_jacket & has_outer & no_base, ['outer_layer']] = 'None'
        df3.loc[no_jacket & no_outer & has_base, ['outer_layer']] = df['base_layer']
        df3.loc[no_jacket & no_outer & has_base, ['base_layer']] = 'None'
        return df3

    def get_dataframe_format(self):

        df = self.ingest_data().copy()
        return df.truncate(after=-1)

    def add_to_sample_data(self, forecast, outfit, athlete_name='default', **kwargs):
        """
        Add a row of sample data to a model file.
        :param forecast:
        :param outfit:
        :param athlete_name:
        :return:
        """

        # Check to see if we have a dataframe already
        # If there is no dataframe, then create one
        if self._sample_data is None:
            self._sample_data = self.get_dataframe_format()

        # Add the required lines to the dataframe from the supplied info
        forecast = vars(forecast) if type(forecast) is not dict else forecast
        fields = {x: outfit[x] for x in self.labels if x in outfit}
        fields.update({x: forecast[x] for x in self.features if x in forecast})
        fields.update({x: kwargs[x] for x in self.features + self.labels if x in kwargs})
        fields.update({'Athlete': athlete_name, 'activity': self.activity_name})
        record_number = len(self._sample_data)
        for k, v in fields.items():
            self._sample_data.loc[record_number, k] = v
        self._have_sample_data = True
        return self._sample_data

    def write_sample_data(self, filename=""):
        """
        Write the sample data to the file specified if no file specified then a random file name
        :param filename:
        :return: the filename of the where the data was written
        """
        default_file_name = f'outfit_data_{NOW.day}{NOW.month}{NOW.hour}{NOW.minute}.csv'
        fn = get_data_path(default_file_name if filename == "" else filename)
        if not self._have_sample_data:
            ReferenceError('No sample data has been created yet.')
        logger.info(f'Writting sample data to {fn}')
        # Take out the first row that we used to format the dataset
        self._sample_data.to_csv(fn, index=False)
        return fn


class RunningOutfitPredictor(BaseOutfitPredictor, RunningOutfitMixin):
    def __init__(self, ):
        """"""
        super(RunningOutfitPredictor, self).__init__()


class BaseOutfitTranslator(BaseActivityMixin):
    """ This class provides for dealing with the outfit components to full sentence replies

    """
    _base_statements: ClassVar[Dict[str, str]] = \
        {'opening': [f'It looks like',
                     f'Oh my,',
                     f'Well,'],
         'weather': [f'the weather should be',
                     f'Weather underground says'],
         'clothing': [f'I suggest wearing',
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

