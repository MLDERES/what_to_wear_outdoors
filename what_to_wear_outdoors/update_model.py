import pandas as pd
import numpy as np
import pickle
import dateutil
import datetime
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import clothing_options
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import os
from utility import DATA_PATH, MODEL_PATH, get_model_filename
from weather_observation import Forecast, Weather

DATA_FILE = os.path.join(DATA_PATH, 'what i wore running.xlsx')

#  Import and clean data
ds_train = pd.read_csv('../data/what i wore running.csv', parse_dates=True, infer_datetime_format=True,
                       converters={'Humidity': p2f}, true_values=['Yes', 'yes', 'y'], false_values=['No', 'no', 'n'])
ds_train.dropna(how='all', inplace=True)
ds_train.rename({'Length of activity (min)': 'duration', 'Wind': 'wind_speed', 'Wind Dir': 'wind_dir',
                 'Distance': 'distance', 'Condition': 'weather_condition', 'Activity': 'activity', 'Light': 'is_light',
                 'Temp': 'temp', 'Feels like': 'feels_like_temp', 'Humidity': 'pct_humidity',
                 'Long Sleeves': 'long_sleeves',
                 'Short Sleeve': 'short_sleeves', 'Arm Warmers': 'arm_warmers', 'SweatShirt': 'sweatshirt',
                 'Jacket': 'jacket',
                 'Tights': 'tights', 'Shorts': 'shorts', 'Gloves': 'gloves', 'Calf Sleeves': 'calf_sleeves',
                 'Shoe Covers': 'shoe_cover', 'Face Cover': 'face_cover'},
                axis='columns', inplace=True)
ds_train['ears_hat'] = ds_train['Ears'] | ds_train['Hat']
ds_train['heavy_socks'] = ds_train['Heavy Socks'].combine(ds_train['Regular Socks'], lambda x, y: x & (not y), False)
ds_train['activity_date'] = pd.to_datetime(ds_train['Date'], format='%d-%b')

ds_train['activity_hour'] = pd.to_datetime(ds_train['Time'], format='%H:%M').dt.hour

# Note we are only going to assume regular or heavy, with reg = True and heavy = False
ds_train.drop(['Date', 'Time', 'Feel', 'Notes', 'Hat', 'Ears', 'Heavy Socks', 'Regular Socks'],
              axis='columns', inplace=True)

# Establish the categorical variables
ds_train['activity_month'] = ds_train['activity_date'].dt.month.astype('category')
ds_train['activity_length'] = pd.cut(ds_train.duration, bins=[0, 31, 61, 121, 720],
                                     labels=['short', 'medium', 'long', 'extra long'])
ds_train['activity'] = ds_train['activity'].astype('category')
ds_train['weather_condition'] = ds_train['weather_condition'].astype('category')
ds_train['wind_dir'] = ds_train['wind_dir'].astype('category')
print(ds_train.describe())
# print(ds_train.dtypes)

# Categorical data column names
CAT_COLS = ['weather_condition', 'is_light', 'wind_dir',
            'activity_month', 'activity_hour', 'activity_length']
NON_CAT_COLS = ['distance', 'duration', 'wind_speed', 'temp', 'feels_like_temp', 'pct_humidity']
ALL_COLS = CAT_COLS + NON_CAT_COLS
PREDICTION_LABELS = ['long_sleeves', 'short_sleeves',  # 'arm_warmers',
                     'sweatshirt', 'jacket', 'gloves', 'shorts', 'tights', 'calf_sleeves',
                     'ears_hat', 'heavy_socks', ]

full_ds = pd.get_dummies(ds_train, columns=CAT_COLS)

# Filtering specfically more information that one athlete has entered.  Others would be interesting
full_ds = full_ds[(full_ds.activity == 'Run') & (full_ds.Athlete == 'Michael')]
full_ds.drop(['Athlete', 'activity', 'activity_date'], axis='columns', inplace=True)
full_ds.drop(['shoe_cover', 'face_cover'], axis='columns', inplace=True)
# print(full_ds)

model = LogisticRegression()
for clothing_option in PREDICTION_LABELS:
    y = np.ravel(full_ds[clothing_option])
    X = full_ds.drop(clothing_option, axis=1)
    model2 = model.fit(X, y)
    model_score = model2.score(X, y)
    print(f'Score for {clothing_option}: {model_score}')
    print(model2.coef_)
    coefficients = pd.DataFrame({X.columns, model2.coef_})
    pickle.dump(model2, open('../models/michael_run.mdl', 'wb'))
    # Save model to the models folder
    #  names for the athlete and the sport
    model_file = get_model_filename(clothing_option)
    pickle.dump(model, open(model_file, 'wb'))


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

# print(ds_train)
