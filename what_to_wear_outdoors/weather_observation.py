import calendar
import random
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import requests
import dotenv
import os
import re
import logging
import datetime as dt
from pandas.io.json import json_normalize

from what_to_wear_outdoors.utility import get_data_path

NOW = dt.datetime.now()

__all__ = ['Observation', 'Weather']

# TODO: Put this back so that we aren't shipping the API key with the code
dotenv.load_dotenv()
WU_API_KEY = os.getenv("WU_API_KEY")
if WU_API_KEY is None:
    raise EnvironmentError(
        "The environment variable WU_API_KEY is required to run this program.")

# So lets just key our observations off of mon_day_year
#  We'll store pretty date for whatever reason
DATE_HOUR = 'hour'
DATE_MONTH = 'mon'
DATE_DAY = 'mday'
DATE_YEAR = 'year'
DAY_OF_WEEK = 'weekday_name'
FORECAST_KEY = 'hourly_forecast'
FCAST_TIME_KEY = 'FCTTIME'
NULL_VALUE = -9999


class FctKeys:
    """
    This class provides a common set of column names to be used in dictionaries, dataframes and other places
    so that there is one place to update the string and it will be hit everywhere
    """
    HUMIDITY = 'pct_humidity'
    FEEL_TEMP = 'feels_like'
    WIND_SPEED = 'wind_speed'
    WIND_DIR = 'wind_dir'
    PRECIP_PCT = 'pct_precip'
    REAL_TEMP = 'temp_f'
    HEAT_IDX = 'heat_index'
    CONDITION = 'condition'
    column_types = {CONDITION: str, HUMIDITY: float, PRECIP_PCT: float,
                    FEEL_TEMP: float, REAL_TEMP: float, WIND_DIR: str,
                    WIND_SPEED: float}


class Location:
    """
    Encapsulates a possible location for querying from Weather Underground
    """
    def __init__(self, location_name):
        """

        :param location_name:
        """
        # Deal with the case where we have gotten a location rather than a string
        if isinstance(location_name, Location):
            self.__dict__ = deepcopy(location_name.__dict__)
        elif isinstance(location_name, str):
            # Need to determine if this is City, ST - a zipcode or Lat/Long
            expression = r'(^\d{5}$)|(^[\w\s]+),\s*(\w{2}$)|(^[\w\s]+)'
            mo = re.match(expression, str(location_name))
            self._zipcode = self._city = self._state = None
            if mo and mo.group(2) is not None:
                # if we have matched City, State then we need to build the query as ST/City.json
                # otherwise we can just use City or Zip + .json
                self._city = str.strip(mo.group(2))
                self._state = str.strip(mo.group(3))
            else:
                self._zipcode = str.strip(location_name)
        else:
            raise ValueError('location_name must be either a string or a Location object')

    def repl(self, sep='/'):
        """
        Get the representation of this object using the appropriate separator for the city and state (if required)
        :param sep: The separator to use between the city and the state fields if they exist
        :return: either the zip code or the string represented by the state followed by the separator then the city
        """
        return self._zipcode if self._zipcode else sep.join([self._state, self._city])

    @property
    def name(self):
        return self._zipcode if self._zipcode else ','.join([self._city, self._state])

    def __str__(self):
        return self.name

    def __repr__(self):
        return '{0} ({1})'.format(object.__repr__(self), str(self))

class JsonDictionary(dict):

    def __missing__(self, key):
        return None

    def read_int(self, key):
        return None if self[key] is None or int(self[key]) <= NULL_VALUE else int(self[key])

    def read_float(self, key):
        return None if float(self[key]) <= NULL_VALUE else float(self[key])


class Observation:

    def __init__(self):
        super(Observation, self).__init__()
        self.location = ""
        self.condition = ""
        self.feels_like = self.heat_index = self.temp_f = 0
        self.wind_dir = None
        self.wind_speed = 0
        self.precip_chance = 0
        self.tod = 5
        self.month_day = 1
        self.mth = 1
        self.dow = calendar.MONDAY
        self.civil_time = ""
        self.is_light = True
        self.pct_humidity = 50
        self.timestamp = dt.datetime(year=NOW.year, month=self.mth, day=self.month_day, hour=self.tod)

    def __str__(self):
        return f'Observation for {self.location} ' \
            f'on {self.dow} (Month-Day): {self.mth}-{self.month_day} ' \
            f'at {self.civil_time}. ' \
            f'\n\tTemperature (feels like): {self.feels_like} °F' \
            f'\n\tWind {self.wind_speed} mph from {self.wind_dir}' \
            f'\n\tConditions: {self.condition} ' \
            f'\n\tChance of precipitation: {self.precip_chance} %' \
            f'\n\tHumidity: {self.pct_humidity}'

    @staticmethod
    def get_fct_key(d=0, m=0, h=0):
        return "{}_{}_{}".format(m, d, h)

    @staticmethod
    def _read_int(i):
        return None if int(i) <= NULL_VALUE else int(i)

    @staticmethod
    def _read_float(f):
        return None if float(f) <= NULL_VALUE else float(f)

    def from_dict(self, dct, units='english'):
        fcast = Observation()
        units = 'english' if units != 'metric' else 'metric'

        time_dct = JsonDictionary(dct[FCAST_TIME_KEY])

        fcast.tod = time_dct.read_int(DATE_HOUR)
        fcast.mth = time_dct.read_int(DATE_MONTH)
        fcast.dow = time_dct[DAY_OF_WEEK]
        fcast.civil_time = time_dct['civil']
        fcast.month_day = time_dct.read_int(DATE_DAY)
        fcast.timestamp = dt.datetime(year=NOW.year, month=fcast.mth, day=fcast.month_day, hour=fcast.tod)

        fcast.condition_code = dct['condition']  # this would be the technical description
        fcast.condition = dct['wx']  # this is the human readable condition
        fcast.feels_like = self._read_float(dct['feelslike'][units])
        fcast.heat_index_f = self._read_float(dct['heatindex'][units])
        fcast.temp_f = self._read_float(dct['temp'][units])
        fcast.wind_dir = dct['wdir']['dir']  # but then we need the 'dir' or 'degrees' key to get it
        fcast.wind_speed = self._read_float(dct['wspd'][units])
        fcast.precip_chance = self._read_float(dct['pop'])
        fcast.pct_humidity = self._read_float(dct['humidity'])

        return fcast


class Weather:

    def __init__(self):
        self._past_observations = None
        self._forecast_observations = None

    @property
    def past_observations(self):
        return self._past_observations

    @property
    def forecast_observations(self):
        return self._forecast_observations

    @staticmethod
    def random_forecast() -> Observation:
        """ Get a random forecast.

            Used for building up the dataset

        :return: a Observation
        """
        f = Observation()
        f.tod = random.randrange(5, 21)
        f.mth = random.randrange(1, 12)
        f.month_day = random.randrange(1, 28)
        f.timestamp = dt.datetime(year=NOW.year, month=f.mth, day=f.month_day, hour=f.tod)

        f.dow = calendar.day_name[f.timestamp.weekday()]
        f.civil_time = f.timestamp.strftime('%I:%M %p')

        f.pct_humidity = round(random.normalvariate(55, 20))
        f.temp_f = f.feels_like = round(random.normalvariate(55, 20))

        pchance = random.randrange(0, 100, step=10)
        f.precip_chance = 0 if pchance < 20 else (100 if pchance > 80 else pchance)
        # We should have a wide-range but between 0-30mph
        f.wind_speed = min(30, max(0, round(random.normalvariate(8, 8))))
        f.wind_dir = random.choice(list(WIND_DIRECTION.keys()))
        f.condition = random.choices(WEATHER_CONDITIONS,
                                     weights=Weather._get_random_weather_condition_weights())
        return f

    @staticmethod
    def _build_observation_df(dct, location: Location = None):
        """
        Converts the JSON returned from Weather Underground into a dataframe of observations with a date/time index
        :param dct:
        :param location:
        :return:
        """

        forecasts = {}
        if FORECAST_KEY in dct:
            for f in dct[FORECAST_KEY]:
                time_dct = JsonDictionary(f[FCAST_TIME_KEY])
                f_key = Observation.get_fct_key(d=time_dct.read_int(DATE_DAY),
                                                m=time_dct.read_int(DATE_MONTH),
                                                h=time_dct.read_int(DATE_HOUR))
                fct = Observation()
                forecasts[f_key] = fct.from_dict(f)
                forecasts[f_key].location = location
        return forecasts

    def get_weather(self, location_name: Union[str, Location] = '72712', when: dt.datetime = NOW) -> Observation:
        """
        Get the weather observation for a given location and a given day/time.  May be either a forecast (future) or
        an historical observation.
        :param when: the day and time for when the weather should be queried, if it's in the future, then
        we'll get the data for the next ten days and cache it, if it's in the past then we'll get the weather for
        the entire day and keep that in the cache instead
        :type location_name: string
        :param location_name: location should be either a zip code or a city, state abbreviation
        :return: a Observation object
        """
        # dt should be the date and the time
        location = Location(location_name)
        logging.debug('get_weather location = ' + location.name)
        logging.debug('date time = {}'.format(when))
        # Ensuring that we are setting our time to hour precision
        when = dt.datetime(when.year, when.month, when.day, when.hour)
        # Need to determine if we have a forecast or observation for the day in question
        is_fct_request = when > NOW

        """
        If we are looking for a forecast -
            Look in the data folder for a file with the name location_forecast.pkl
            If the file doesn't exist or has a timestamp that is more than 12 hours old -
                Query the weather service
                Save the forecast from the weather service as location_forecast.pkl
                Return the row for the time requested
            If the file does exist and the timestamp is within 12 hours
                Unpickle the file into a dataframe
                If we can find the forecast for the time we want is in the dataframe
                    Return the row for the time requested
                Else 
                    Query the weather service
                    Update the forecast from the weather service as location_forecast.pkl
                    Return the row for the time requested
        
        If we are looking for historical data -
            Look in the data folder for a file with the name location_historic_weather.pkl
            If the file doesn't exist
                Query the weather service for the historical day in question
                Save the historical data to the file location_historic_weather.pkl
                Return the row for the time requested
            If the file does exist
                Unpickle the file into a dataframe
                Query for the date in question
                If the date doesn't exist
                    Query the weather service for the day in question
                    Append to the unpickled dataframe
                    Re-pickle the data frame and save to the data folder
                    Return the row for the time requested
        
        Convert the series, returned from either of the two function calls into an Observation object
        """
        working_df = self.forecast_observations if is_fct_request else self.past_observations

        if working_df is None:  # TODO: Or we can't find a suitable answer in our cache
            # We don't even have a dataframe, so we need to create one
            weather_request = Weather._build_forecast_request(location) if is_fct_request \
                else Weather._build_historical_request(location, when)
            resp = requests.get(weather_request)
            if resp.status_code == 200:
                wu_response = resp.json()
                if is_fct_request:
                    working_df = self._forecast_observations = self._build_forecast_df(wu_response, location)
                else:
                    working_df = self._past_observations = self._build_historic_df(wu_response, location)
            else:
                raise ValueError(f'Danger Will Robinson we got a bad response from WU {resp.status_code}')
        return working_df.loc[when]

    def _get_weather_forecast(self, location, fct_ts) -> pd.Series:
        """
        Return the forecast for a given date/time
        Look in the data folder for a file with the name location_forecast.pkl
            If the file doesn't exist or has a timestamp that is more than 12 hours old -
                Query the weather service
                Save the forecast from the weather service as location_forecast.pkl
                Return the row for the time requested
            If the file does exist and the timestamp is within 12 hours
                Unpickle the file into a dataframe
                If we can find the forecast for the time we want is in the dataframe
                    Return the row for the time requested
                Else
                    Query the weather service
                    Update the forecast from the weather service as location_forecast.pkl
                    Return the row for the time requested
        :param location: a valid location (either
        :return: a pd.Series that has all the info needed to build an Observation

        """
        fct_file = get_data_path(self.fct_filename(location))

    @staticmethod
    def _build_forecast_request(location):
        # http://api.wunderground.com/api/7d65568686ff9c25/features/settings/q/query.format
        # Features = alerts/almanac/astromony/conditions/forecast/hourly/hourly10day etc.
        # settings(optional) = lang, pws(personal weather stations):0 or 1
        # query = location (ST/City, zipcode,Country/City, or lat,long)
        # format = json or xml
        loc = Location(location)
        request = f'http://api.wunderground.com/api/{WU_API_KEY}/hourly10day/q/' \
            f'{loc.repl("/")}.json'
        return request

    @staticmethod
    def _build_historical_request(location, d):
        # http://api.wunderground.com/api/7d65568686ff9c25/features/settings/q/query.format
        # Features = alerts/almanac/astromony/conditions/forecast/hourly/hourly10day etc.
        # settings(optional) = lang, pws(personal weather stations):0 or 1
        # query = location (ST/City, zipcode,Country/City, or lat,long)
        # format = json or xml
        rdate = d.strftime('%Y%m%d')
        loc = Location(location)
        request = f'http://api.wunderground.com/api/{WU_API_KEY}/history_{rdate}/q/' \
            f'{loc.repl("/")}.json'
        return request

    @staticmethod
    def _build_forecast_df(weather_json, location):
        df = json_normalize(weather_json['hourly_forecast']) \
            .drop(columns=['FCTTIME.UTCDATE', 'FCTTIME.age', 'FCTTIME.ampm', 'FCTTIME.civil',
                           'FCTTIME.epoch', 'FCTTIME.hour_padded', 'FCTTIME.isdst', 'FCTTIME.min',
                           'FCTTIME.mday_padded', 'FCTTIME.min_unpadded', 'FCTTIME.mon_abbrev',
                           'FCTTIME.mon_padded', 'FCTTIME.month_name', 'FCTTIME.month_name_abbrev',
                           'FCTTIME.pretty', 'FCTTIME.sec', 'FCTTIME.tz', 'FCTTIME.weekday_name',
                           'FCTTIME.weekday_name_abbrev', 'FCTTIME.weekday_name_night',
                           'FCTTIME.weekday_name_night_unlang', 'FCTTIME.weekday_name_unlang',
                           'FCTTIME.yday', 'heatindex.english', 'heatindex.metric', 'icon', 'icon_url',
                           'mslp.english', 'mslp.metric', 'qpf.english', 'qpf.metric', 'sky',
                           'snow.english', 'snow.metric', 'uvi', 'wdir.degrees', 'windchill.english',
                           'windchill.metric', 'temp.metric', 'feelslike.metric', 'wspd.metric',
                           'fctcode', 'dewpoint.english', 'dewpoint.metric', 'wx']) \
            .rename(
            columns={'FCTTIME.mday': 'day', 'FCTTIME.hour': 'hour', 'FCTTIME.mon': 'month', 'FCTTIME.year': 'year',
                     'temp.english': FctKeys.REAL_TEMP, 'feelslike.english': FctKeys.FEEL_TEMP,
                     'wdir.dir': FctKeys.WIND_DIR, 'wspd.english': FctKeys.WIND_SPEED, 'conds': FctKeys.CONDITION,
                     'humidity': FctKeys.HUMIDITY, 'pop': FctKeys.PRECIP_PCT, 'wirde': FctKeys.WIND_DIR})
        dt_index = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.set_index(dt_index, inplace=True)
        df.drop(columns=['hour', 'day', 'month', 'year'], inplace=True)
        df = df.astype(FctKeys.column_types)
        df['location'] = location
        return df

    @staticmethod
    def _build_historic_df(weather_json, location):
        """
        From an historic weather observation (for a given day) build a dataframe with the weather data
        :param weather_json: The response from the Weather Underground historic weather request
        :param location: a location to associate this observation
        :return: a dataframe
        """
        df = json_normalize(weather_json['history'], 'observations') \
            .drop(columns=['icon', 'metar', 'utcdate', 'wgustm', 'wgusti', 'wdird', 'vism', 'visi', 'pressurem',
                           'pressurei', 'tornado', 'fog', 'hail', 'thunder', 'dewpti', 'dewptm', 'precipm',
                           'wspdm', 'windchillm', 'tempm', 'heatindexm', 'precipi']) \
            .rename(columns={'conds': FctKeys.CONDITION, 'hum': FctKeys.HUMIDITY, 'tempi': FctKeys.REAL_TEMP,
                             'wdire': FctKeys.WIND_DIR, 'wspdi': FctKeys.WIND_SPEED})

        df = df.astype({'heatindexi': float, 'windchilli': float})
        df[FctKeys.FEEL_TEMP] = np.where(df['heatindexi'] == NULL_VALUE, df.windchilli, df.heatindexi)
        df.drop(columns=['heatindexi', 'windchilli'], inplace=True)
        df[FctKeys.PRECIP_PCT] = np.where((df['rain'] == 1), 100, 0)
        df[FctKeys.PRECIP_PCT] = np.where((df['snow'] == 1), 100, df[FctKeys.PRECIP_PCT])
        df.drop(columns=['rain', 'snow'], inplace=True)
        df = df.astype(FctKeys.column_types)

        df2 = json_normalize(df['date']).rename(columns={'mon': 'month', 'mday': 'day', 'min': 'minute'}) \
            .drop(columns=['pretty', 'tzname'])
        df.drop(columns=['date'], inplace=True)

        df3 = pd.to_datetime(df2).values.astype('datetime64[h]')
        df.set_index(df3, inplace=True)
        df['location'] = location
        return df

    @staticmethod
    def _get_random_weather_condition_weights():
        return (([70] * len(_high_conditions)) +
                ([20] * len(_med_high_conditions)) +
                ([6] * len(_med_conditions)) +
                ([3] * len(_low_conditions)) +
                ([1] * len(_unlikely_conditions)))

    @staticmethod
    def fct_filename(location):
        """
        Returns the filename for a future forecast for a given location
        :param location:
        :return:
        """
        loc = Location(location)
        return f'{loc.repl("_")}_forecast.weather'

    @staticmethod
    def obs_filename(location):
        """
        Returns the filename of the historical weather observations for a provided location
        :param location:
        :return:
        """
        loc = Location(location)
        return f'{loc.repl("_")}_historical.weather'


_All_Forecasts_Location = JsonDictionary()

_high_conditions = ['Clear', 'Overcast', 'Partly Cloudy', 'Scattered Clouds', 'Mostly Cloudy', ]
_med_high_conditions = ['Partial Fog', 'Patches of Fog', 'Light Rain', ]
_med_conditions = ['Unknown', 'Heavy Thunderstorms and Rain', 'Heavy Fog', 'Heavy Rain', 'Heavy Rain Showers',
                   'Heavy Fog Patches', 'Heavy Mist', 'Heavy Rain Mist',
                   'Heavy Thunderstorm', 'Light Thunderstorms and Rain', 'Light Fog', 'Light Fog Patches',
                   'Light Rain Showers', 'Light Mist', 'Light Rain Mist', 'Light Thunderstorm', ]
_low_conditions = ['Heavy Snow', 'Heavy Drizzel', 'Heavy Hail Showers', 'Heavy Haze', 'Heavy Snow Showers',
                   'Light Snow', 'Light Drizzel', 'Light Hail Showers', 'Light Haze', 'Light Snow Showers', ]
_unlikely_conditions = ['Heavy Freezing Drizzle', 'Heavy Freezing Fog', 'Heavy Freezing Rain', 'Heavy Hail',
                        'Heavy Small Hail Showers', 'Heavy Snow Blowing Snow Mist',
                        'Heavy Snow Grains', 'Heavy Thunderstorms and Snow', 'Heavy Thunderstorms with Hail',
                        'Heavy Thunderstorms with Small Hail', 'Heavy Thunderstorms and Ice Pellets', 'Heavy Spray',
                        'Light Freezing Drizzle', 'Light Freezing Fog', 'Light Freezing Rain', 'Light Hail',
                        'Light Small Hail Showers', 'Light Snow Blowing Snow Mist', 'Light Snow Grains',
                        'Light Thunderstorms and Snow', 'Light Thunderstorms with Hail',
                        'Light Thunderstorms with Small Hail', 'Light Thunderstorms and Ice Pellets', 'Light Spray',
                        'Shallow Fog', 'Small Hail', 'Squalls', 'Unknown Precipitation']
WEATHER_CONDITIONS = _high_conditions + _med_high_conditions + _med_conditions + _low_conditions + _unlikely_conditions

WIND_DIRECTION = {'N': 'North', 'NNE': 'North-Northeast', 'NNW': 'North-Northwest',
                  'ENE': 'East-Northeast', 'WNW': 'West-Northwest', 'E': 'East', 'W': 'West',
                  'S': 'South', 'SSE': 'South-Southeast', 'SSW': 'South-Southwest',
                  'ESE': 'East-Southeast', 'WSW': 'West-Southwest'}
