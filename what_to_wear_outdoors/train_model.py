import pandas as pd
from pandas.api.types import CategoricalDtype
import logging
import pickle
import datetime
from sklearn.linear_model import LogisticRegression
from pathlib import Path

if __package__ == '' or __name__ == '__main__':
    from utility import get_model, get_data, get_model_name
else:
    from .utility import get_model, get_data, get_model_name


def train(sport='Run'):
    """ Create the data models from the input file specified by the filename and filepath

    :param filename: name of the xlsx file where the data is stored
    :param filepath: path to find the data file, if ''
    :return: None
    """
    logger = logging.getLogger(__name__)
    data_file = get_data('what i wore running.xlsx')
    logger.debug(f'Reading data from {data_file}')
    #TODO: Check for valid values of sport in the request to train
    valid_sports = ['Run']
    sport = 'Run'
    logger.warning(f'The only sports that are supported at this point are {valid_sports}')
    #  Import and clean data
    full_ds = pd.read_excel(data_file, sheet_name='Activity Log',
                            skip_blank_lines=True,
                            true_values=['Yes', 'yes', 'y'], false_values=['No', 'no', 'n'],
                            dtype={'Time': datetime.time})
    full_ds.dropna(how='all', inplace=True)
    full_ds.fillna(value=False, inplace=True)
    logger.debug(f'Data file shape: {data_file}')

    full_ds.rename({'Length of activity (min)': 'duration', 'Wind': 'wind_speed', 'Wind Dir': 'wind_dir',
                    'Date': 'activity_date', 'Distance': 'distance', 'Condition': 'weather_condition',
                    'Activity': 'activity', 'Light': 'is_light', 'Temp': 'temp', 'Feels like': 'feels_like_temp',
                    'Humidity': 'pct_humidity',
                    'Long Sleeves': 'long_sleeves', 'Short Sleeve': 'short_sleeves', 'Arm Warmers': 'arm_warmers',
                    'SweatShirt': 'sweatshirt', 'Jacket': 'jacket', 'Tights': 'tights', 'Shorts': 'shorts',
                    'Gloves': 'gloves', 'Calf Sleeves': 'calf_sleeves', 'Shoe Covers': 'shoe_cover',
                    'Face Cover': 'face_cover', 'Heavy Socks': 'heavy_socks', 'Time': 'activity_hour'},
                   axis='columns', inplace=True)

    # Now deal with the special cases
    full_ds['ears_hat'] = full_ds['Ears'] | full_ds['Hat']
    full_ds.drop(['Feel', 'Notes', 'Hat', 'Ears', 'activity_hour'], axis='columns', inplace=True, errors='ignore')

    # Establish the categorical variables
    full_ds['activity_month'] = full_ds['activity_date'].dt.month.astype('category')
    full_ds['activity_length'] = pd.cut(full_ds.duration, bins=[0, 31, 61, 121, 720],
                                        labels=['short', 'medium', 'long', 'extra long'])
    full_ds['activity'] = full_ds['activity'].astype('category')
    full_ds['weather_condition'] = full_ds['weather_condition'].astype('category')
    full_ds['wind_dir'] = full_ds['wind_dir'].astype('category')
    tights_cat_type = CategoricalDtype(categories=["None", "Capri", "Long", "Insulated"], ordered=True)
    full_ds['tights'] = full_ds['tights'].astype(tights_cat_type)

    # Categorical data column names
    # TODO: When we get good enough add in a few more categorical variables
    CAT_COLS = ['is_light']  # , 'activity_length','wind_dir', 'weather_condition','activity_month'

    NON_CAT_COLS = ['duration', 'wind_speed', 'feels_like_temp', 'pct_humidity']  # 'distance','temp',
    ALL_FACTOR_COLS = CAT_COLS + NON_CAT_COLS
    PREDICTION_LABELS = ['long_sleeves', 'short_sleeves', 'sweatshirt', 'jacket', 'gloves', 'shorts', 'tights',
                         'calf_sleeves', 'ears_hat', 'heavy_socks',
                         # Removing for now as they haven't yet been used for running
                         # 'arm_warmers','face_cover','shoe_cover',
                         ]

    # TODO: Deal with the case where a particular column has no value in terms of that sport.
    #   In other words, if shoe covers is always 0 for Running,then don't bother to predict shoe covers and just assume
    #   that they aren't necessary
    def check_useless_columns():
        useless_columns = {}
        for l in PREDICTION_LABELS:
            if (full_ds[l].mean == 0) or (full_ds[l].mean == 1):
                useless_columns[l] = full_ds[l].mean

    # TODO: --SPORT FILTER -- Handle the case where the activity and athlete are variable
    # Filtering specifically more information that one athlete has entered.  Others would be interesting
    full_train_ds = full_ds[(full_ds.activity == sport)].copy()
    # TODO: --SPORT FILTER -- Remove this when we have taken care to handle multiple athletes
    full_train_ds.drop(['Athlete', 'activity', 'activity_date'], axis='columns', inplace=True)
    # TODO: --SPORT FILTER -- This can go when we have the filter by sport - for columns that don't change values
    full_train_ds.drop(['shoe_cover', 'face_cover', 'arm_warmers'], axis='columns', inplace=True)

    train_ds = full_train_ds[ALL_FACTOR_COLS]
    train_ds = pd.get_dummies(train_ds, columns=CAT_COLS)

    # Finally, we are going to be able to establish the model.
    # TODO: Does it make sense to put the prediction labels in order such that once we predict a label it becomes part
    #   of the model for subsequent models?
    # TODO: This can be done by creating a multivariate logistic regression or by doing a cross-product of all the
    #   combinations and then using these combinations as a single multi-class regression predictor
    X = train_ds
    for clothing_option in PREDICTION_LABELS:
        logger.info('Building model for {clothing_option}')
        model = LogisticRegression(solver='lbfgs')
        y = full_train_ds[clothing_option]
        model.fit(X, y)
        model_score = model.score(X, y)
        logger.debug(f'Score for {clothing_option}: {model_score}')
        c = list(zip(list(X), model.coef_[0]))
        coefficients = pd.DataFrame(c)
        # logging.debug(coefficients)
        logger.debug(f'{clothing_option}\n{coefficients}')
        # Save model to the models folder
        #  names for the athlete and the sport
        model_file = get_model(get_model_name(clothing_option))
        pickle.dump(model, open(model_file, 'wb'))
        logger.info(f'Model written to {model_file}')

    '''
    Columns as read from data file
    Index(['Athlete', 'Date', 'Time', 'Activity', 'Distance',
           'Length of activity (min)', 'Condition', 'Light', 'Wind', 'Wind Dir',
           'Temp', 'Feels like', 'Humidity', 'Feel', 'Notes', 'Hat', 'Ears',
           'Long Sleeves', 'Short Sleeve', 'Arm Warmers', 'SweatShirt', 'Jacket',
           'Gloves', 'Shorts', 'Tights', 'Calf Sleeves', 'Regular Socks',
           'Heavy Socks', 'Shoe Covers', 'Face Cover'],
    
    # Post data cleaning factors
    activity                   category
    distance                    float64
    duration                      int64
    weather_condition          category
    is_light                       bool
    wind_speed                    int64
    wind_dir                   category
    temp                          int64
    feels_like_temp               int64
    pct_humidity                float64
    long_sleeves                   bool
    short_sleeves                  bool
    arm_warmers                    bool
    sweatshirt                     bool
    jacket                         bool
    gloves                         bool
    shorts                         bool
    tights                         bool
    calf_sleeves                   bool
    shoe_cover                     bool
    face_cover                     bool
    ears_hat                       bool
    heavy_socks                    bool
    activity_date        datetime64[ns]
    activity_hour                 int64
    activity_month             category
    activity_length            category
    '''


if __name__ == '__main__':
    # create logger with 'spam_application'
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[1]
    train()
