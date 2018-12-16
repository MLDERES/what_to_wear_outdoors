import pandas as pd
from pandas.api.types import CategoricalDtype
import logging
import pickle
import datetime
from pathlib import Path
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

if __package__ == '' or __name__ == '__main__':
    from utility import get_model, get_data, get_model_name
else:
    from .utility import get_model, get_data, get_model_name

logger = logging.getLogger(__name__)


def train(sport='Run'):
    """ Create the data models from the input file specified by the filename and filepath

    :param sport:
    :return: None

    Athlete    object
    Date    datetime64[ns]
    Time    object
    Activity    object
    Distance    float64
    Length of activity(min)    int64
    Condition    object
    Light    bool
    Wind    int64
    Wind Dir   object
    Temp    int64
    Feels like int64
    Humidity    float64
    Feel    object
    Notes    object
    Hat - Ears    bool
    Outer Layer    object
    Base Layer    object
    Arm Warmers   bool
    Jacket    object
    Gloves    bool
    LowerBody    object
    Heavy Socks  bool
    Shoe Covers  bool
    Face Cover   bool
    Unnamed: 25  float64
    Unnamed: 26  object
    Unnamed: 27 float64
    """
    data_file = get_data('what i wore running.xlsx')

    logger.debug(f'Reading data from {data_file}')
    # TODO: Check for valid values of sport in the request to train
    valid_sports = ['Run']
    logger.info(f'The only sports that are supported at this point are {valid_sports}')
    #  Import and clean data
    full_ds = pd.read_excel(data_file, sheet_name='Activity Log2',
                            skip_blank_lines=True,
                            true_values=['Yes', 'yes', 'y'], false_values=['No', 'no', 'n'],
                            dtype={'Time': datetime.time}, usecols='A:Y')
    full_ds.dropna(how='all', inplace=True)
    full_ds.fillna(value=False, inplace=True)
    logger.debug(f'Data file shape: {data_file}')

    full_ds.rename({'Date': 'activity_date', 'Time': 'activity_hour', 'Activity': 'activity', 'Distance': 'distance',
                    'Length of activity (min)': 'duration', 'Condition': 'weather_condition', 'Light': 'is_light',
                    'Wind': 'wind_speed', 'Wind Dir': 'wind_dir', 'Temp': 'temp', 'Feels like': 'feels_like_temp',
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

    leg_categories = ['Shorts', 'Shorts-calf cover', 'Capri', 'Long tights', 'Insulated tights']
    layer_categories = ['None', 'Sleeveless', 'Short-sleeve', 'Long-sleeve', 'Sweatshirt-Heavy']
    full_ds['lower_body'] = full_ds['lower_body'].astype(CategoricalDtype(categories=leg_categories, ordered=True))
    full_ds['outer_layer'] = full_ds['outer_layer'].astype(CategoricalDtype(categories=layer_categories, ordered=True))
    full_ds['base_layer'] = full_ds['base_layer'].astype(CategoricalDtype(categories=layer_categories, ordered=True))

    # Categorical data column names
    # TODO: When we get good enough add in a few more categorical variables
    CAT_COLS = ['is_light']  # , 'activity_length','wind_dir', 'weather_condition','activity_month'

    NON_CAT_COLS = ['duration', 'wind_speed', 'feels_like_temp', 'pct_humidity'] #,'distance' 'temp']
    ALL_FACTOR_COLS = NON_CAT_COLS + CAT_COLS
    PREDICTION_LABELS = ['ears_hat', 'outer_layer', 'base_layer',
                         'jacket', 'gloves', 'lower_body','heavy_socks',
                         #'arm_warmers', 'face_cover', 'shoe_cover' - not used for running
                         ]

    full_train_ds = full_ds[(full_ds.activity == sport)].copy()
    # TODO: --SPORT FILTER -- Remove this when we have taken care to handle multiple athletes
    full_train_ds.drop(['Athlete', 'activity', 'activity_date'], axis='columns', inplace=True)
    # TODO: --SPORT FILTER -- This can go when we have the filter by sport - for columns that don't change values
    full_train_ds.drop(['shoe_cover', 'face_cover', 'arm_warmers'], axis='columns', inplace=True)

    train_ds = full_train_ds[ALL_FACTOR_COLS]
    print(f'{train_ds.dtypes}')
    # Finally, we are going to be able to establish the model.
    # TODO: Does it make sense to put the prediction labels in order such that once we predict a label it becomes part
    #   of the model for subsequent models?
    # TODO: This can be done by creating a multivariate logistic regression or by doing a cross-product of all the
    #   combinations and then using these combinations as a single multi-class regression predictor
    X = train_ds
    y_bools = full_train_ds[['ears_hat', 'gloves', 'heavy_socks']]
    y_class = full_train_ds[['outer_layer', 'base_layer', 'jacket', 'lower_body']]

    # https://scikit-learn.org/stable/modules/multiclass.html
    # We have to do two models, one for booleans and one for classifiers (which seems dumb)
    # Starting with the classifiers
    forest = DecisionTreeClassifier(max_depth=4)
    mt_forest = MultiOutputClassifier(forest, n_jobs=-1)
    mt_forest.fit(X, y_class)
    pred_X = [[30, 10, 25, .8, True], [40, 10, 60, .8, True]]
    print(f'Outer Base Jacket Lower {mt_forest.predict(pred_X)}')
    model_file = get_model(get_model_name('classifiers'))
    pickle.dump(mt_forest, open(model_file, 'wb'))

    # Now for the booleans
    forest = DecisionTreeClassifier(max_depth=4)
    mt_forest = MultiOutputClassifier(forest)
    mt_forest.fit(X, y_bools)
    pred_X = [[30, 10, 25, .8, True], [40, 10, 60, .8, True], [80, 0, 75, .3, False]]
    model_file = get_model(get_model_name('booleans'))
    pickle.dump(mt_forest, open(model_file, 'wb'))

    #['duration', 'wind_speed', 'feels_like_temp', 'pct_humidity', 'is_light']
    # pred_X = [[30, 10, 25, .8, True], [40, 10, 60, .8, True]]
    print(f'Ears, Gloves, Heavy Socks {mt_forest.predict(pred_X)}')

    # mt_forest.fit(X, y2)
    # print(f'Outer, Base, Jacket, Lower Body {mt_forest.predict(pred_X)}')

class predictor():
    '''
    The idea here is to be able to pickle a predictor with both ML models (for classification and boolean)
    Each predictor will hold the model, the input parameters to the model and the prediction labels for each
    model type

    I expect that I should be able to:
        1. determine if a model is out of date - is_model_latest
        2. inflate a model from a file - inflate_model
        3. run both models against a set of input and return a dictionary of outputs
        4. update a model from source data if it is out of date
        5. pickle itself and unpickle so that we can use the functions within  
    '''

    @property
    def boolean_predictor(self):
        return

    @boolean_predictor.setter
    def boolean_predictor(self, value):
        pass

    @property
    def classifier(self):
        return

    @classifier.setter
    def classifier(self, value):
        pass

if __name__ == '__main__':
    # create logger with 'spam_application'
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[1]
    train()
