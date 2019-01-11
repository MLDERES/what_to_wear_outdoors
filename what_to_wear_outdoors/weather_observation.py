import calendar
import json
import pickle
import random
from pathlib import Path
from typing import Union

import requests
import dotenv
import os
import logging
import datetime as dt

from what_to_wear_outdoors.location import Location
from what_to_wear_outdoors.utility import get_data_path, is_file_newer, read_float, read_int

NOW = dt.datetime.now()

__all__ = ['Observation', 'Weather']

# TODO: Put this back so that we aren't shipping the API key with the code
dotenv.load_dotenv()
WU_API_KEY = os.getenv("WU_API_KEY")
DARK_SKY_KEY = os.getenv('DARK_SKY_KEY')
if DARK_SKY_KEY is None:
    raise EnvironmentError(
        "The environment variable DARK_SKY_KEY is required to run this program.")

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
    WIND_CHILL = 'wind_chill'
    HUMIDITY = 'pct_humidity'
    FEEL_TEMP = 'feels_like'
    WIND_SPEED = 'wind_speed'
    WIND_DIR = 'wind_dir'
    PRECIP_PCT = 'pct_precip'
    REAL_TEMP = 'temp_f'
    HEAT_IDX = 'heat_index'
    CONDITION = 'condition'
    LOCATION = 'location'
    TIMESTAMP = 'timestamp'
    column_types = {CONDITION: str, HUMIDITY: float, PRECIP_PCT: float,
                    FEEL_TEMP: float, REAL_TEMP: float, WIND_DIR: str, WIND_SPEED: float}


class JsonDictionary(dict):

    def __missing__(self, key):
        return None

    def read_int(self, key):
        return None if self[key] is None or int(self[key]) <= NULL_VALUE else int(self[key])

    def read_float(self, key):
        return None if float(self[key]) <= NULL_VALUE else float(self[key])


class Observation:
    __valid_units = ['imperial', 'english', 'metric']

    def __init__(self, location: [Location, str] = '', timestamp=None, condition='', temp: float = 0,
                 wind_dir: str = None, wind_speed: float = 0, pop: float = 0, pct_humidity: float = 0,
                 wind_chill: float = 0, heat_index: float = 0, is_daylight=True, units: str = 'imperial'):
        assert units in Observation.__valid_units
        super(Observation, self).__init__()
        self.location = Location(location)
        self.units = units
        self.condition = condition
        self.feels_like = heat_index if heat_index != NULL_VALUE else wind_chill
        self.temperature = temp
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
        self.precip_chance = pop
        self.is_light = is_daylight
        self.pct_humidity = pct_humidity if pct_humidity < 1 else (pct_humidity / 100)
        self.timestamp = timestamp

    def __str__(self):
        return f'Observation for {self.location} ' \
            f'on {self.dow} (Month-Day): {self.mth}-{self.month_day} ' \
            f'at {self.civil_time}. ' \
            f'\n\tTemperature (feels like): {self.feels_like} °F' \
            f'\n\tWind {self.wind_speed} mph from {self.wind_dir}' \
            f'\n\tConditions: {self.condition} ' \
            f'\n\tChance of precipitation: {self.precip_chance} %' \
            f'\n\tHumidity: {self.pct_humidity * 100 }%'

    @property
    def dow(self):
        if isinstance(self.timestamp,dt.datetime):
            return calendar.day_name[self.timestamp.weekday()]
        else:
            return None

    @property
    def mth(self):
        if isinstance(self.timestamp, dt.datetime):
            return self.timestamp.month
        else:
            return None

    @property
    def month_day(self):
        if isinstance(self.timestamp, dt.datetime):
            return self.timestamp.day
        else:
            return None

    @property
    def civil_time(self):
        if isinstance(self.timestamp, dt.datetime):
            return self.timestamp.strftime('%H:00')
        else:
            return None

    @staticmethod
    def get_fct_key(d=0, m=0, y=NOW.year):
        return f"{y}_{m}_{d}"

    @staticmethod
    def _read_int(i):
        return None if int(i) <= NULL_VALUE else int(i)

    @staticmethod
    def _read_float(f):
        return None if float(f) <= NULL_VALUE else float(f)


class Weather:

    def __init__(self):
        self._past_observations = None
        self._forecast_observations = None

    @staticmethod
    def calc_wdir(bearing):
        val = int((bearing / 22.5) + .5)
        arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return arr[(val % 16)]

    @staticmethod
    def random_forecast() -> Observation:
        """ Get a random forecast.

            Used for building up the dataset

        :return: a Observation
        """
        ts = dt.datetime(year=NOW.year, month=random.randrange(1, 12),
                         day=random.randrange(1, 28), hour=random.randrange(5, 21))
        pchance = random.randrange(0, 100, step=10)
        pchance = 0 if pchance < 20 else (100 if pchance > 80 else pchance)
        # We should have a wide-range but between 0-30mph
        temp = round(random.normalvariate(55, 20))

        return Observation(location='Anytown, US',
                           timestamp=ts,
                           condition=random.choices(WEATHER_CONDITIONS,
                                                    weights=Weather._get_random_weather_condition_weights()),
                           temp=temp,
                           wind_dir=random.choice(list(WIND_DIRECTION.keys())),
                           wind_speed=min(30, max(0, round(random.normalvariate(8, 8)))),
                           pop=pchance,
                           pct_humidity=round(random.normalvariate(55, 20)),
                           wind_chill=temp if temp < 50 else NULL_VALUE,
                           heat_index=temp if temp >= 50 else NULL_VALUE)

    @staticmethod
    def _build_observations_from_darksky_json(dct, location: Location = None, units='english'):
        """
        Converts the JSON returned from Weather Underground into a dataframe of observations with a date/time index
        :param dct:
        :param location:
        :return:
        """

        forecasts = {}
        fcast = None
        if dct.get('hourly'):
            for f in dct.get('hourly').get('data'):
                ts = dt.datetime.fromtimestamp(f.get('time'))
                f_key = Observation.get_fct_key(y=ts.year, d=ts.day, m=ts.month)
                day_hr = ts.hour

                condition =f.get('icon',"").rstrip('daynight').rstrip('-')
                wind_dir = Weather.calc_wdir(f.get('windBearing',0))
                fcast = Observation(location=location, timestamp=ts, condition=condition,
                                    temp=read_float(f.get('temperature')),
                                    wind_dir=wind_dir, wind_speed=read_float(f.get('windSpeed')),
                                    pop=read_float(f.get('precipProbability')),
                                    pct_humidity=read_float(f.get('humidity')),
                                    heat_index=read_float(f.get('apparentTemperature')),
                                    units=units)

                if f_key not in forecasts.keys():
                    forecasts[f_key] = {}

                forecasts[f_key][day_hr] = fcast
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
        if when > NOW:
            weather_obs = self._get_weather_forecast(location, when)
        else:
            weather_obs = self._get_past_observation(location, when)

        return weather_obs

    def _get_weather_forecast(self, location, fct_ts):
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
        fct_ts = dt.datetime(fct_ts.year, fct_ts.month, fct_ts.day, fct_ts.hour)
        fct_key = Observation.get_fct_key(d=fct_ts.day, m=fct_ts.month, y=fct_ts.year)
        if self._forecast_observations is None:
            logging.debug(f'Didn`t have any forecasts in memory.  Looking to cached file')
            if Path(fct_file).exists() and is_file_newer(fct_file, hours=12):
                # Load the file and read it into a dataframe we can work with
                logging.debug(f'Opening the forecast file for {location} because it`s less than 12 hours old')
                self._forecast_observations = self.load_obj(fct_file)
            else:
                logging.debug(f'Forecast file not found or too old.')
                weather_response = self.get_darksky_weather(location)
                self._forecast_observations = self._build_observations_from_darksky_json(weather_response, location)
                self.save_obj(self._forecast_observations, fct_file)

        fct = self._forecast_observations[fct_key][fct_ts.hour]
        return fct

    def _get_past_observation(self, location, timestamp) -> Observation:
        """
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
        """
        # If the file does exist
        #     Unpickle the file into a dataframe
        #     Query for the date in question
        #     If the date doesn't exist
        #         Query the weather service for the day in question
        #         Append to the unpickled dataframe
        #         Re-pickle the data frame and save to the data folder
        #         Return the row for the time requested
        obs_file = get_data_path(self.obs_filename(location))
        timestamp = dt.datetime(timestamp.year, timestamp.month, timestamp.day, timestamp.hour)
        if self._past_observations is None:  # Must leave this condition with a dataframe
            # We haven't loaded up the file yet, so let's do that.
            # Look in the data folder for a file with the name location_historic_weather.pkl
            if obs_file.exists():
                logging.debug(f'Found a cache file for {location}')
                self._past_observations = self.load_obj(obs_file)
            else:
                logging.debug(f'Didn`t find a cached file for {location}')
                weather_response = self.get_darksky_weather(location, timestamp)
                hist_df = self._build_observations_from_darksky_json(weather_response, location)
                # Since we came in here with no _past_observations we can safely override this value
                self._past_observations = (hist_df)
                self.save_obj(self._past_observations, obs_file)

        # Now we have a dictionary (if not we are going to bail)
        assert self._past_observations is not None
        request_key = Observation.get_fct_key(d=timestamp.day,m=timestamp.month, y=timestamp.year)
        # Check to see if we can find the DAY in our past observations
        if request_key not in self._past_observations.keys():
            # We didn't have the DAY in our list so, we need to go out to the weather service
            weather_response = self.get_darksky_weather(location, timestamp)
            hist_df = self._build_observations_from_darksky_json(weather_response, location)
            self._past_observations.update(hist_df)
            self.save_obj(self._past_observations, obs_file)

        return self._past_observations[request_key][timestamp.hour]

    @staticmethod
    def get_darksky_weather(location, when=None):
        """

        :param location:
        :param when:
        :return:
        """
        loc = Location(location)
        weather_request_url = f'https://api.darksky.net/forecast/{DARK_SKY_KEY}/{loc.lat},{loc.long}'
        if when is not None:
            weather_request_url = f'{weather_request_url},{int(when.timestamp())}'
        logging.debug(f'Going out to Dark Sky for a request for {location} {when}')
        logging.debug(f'{weather_request_url}')
        resp = requests.get(weather_request_url,
                            params={'exclude': ['currently', 'minutely', 'daily', 'alerts', 'flags']})
        if resp.status_code != 200:
            raise ValueError(f'Danger Will Robinson we got a bad response from WU {resp.status_code}')

        return resp.json()

    @staticmethod
    def _get_random_weather_condition_weights():
        return ([70] * len(_high_conditions)) + ([30] * len(_low_conditions))

    @staticmethod
    def fct_filename(location):
        """
        Returns the filename for a future forecast for a given location
        :param location:
        :return:
        """
        loc = Location(location)
        return f'{loc.repl("_")}_forecast.json'

    @staticmethod
    def obs_filename(location):
        """
        Returns the filename of the historical weather observations for a provided location
        :param location:
        :return:
        """
        loc = Location(location)
        return f'{loc.repl("_")}_historical.json'

    @staticmethod
    def save_obj(obj, filename):
        with open(filename,'wb') as f:
            pickle.dump(obj,f)

    @staticmethod
    def load_obj(filename):
        with open(filename,'rb') as f:
            return pickle.load(f)

_All_Forecasts_Location = JsonDictionary()

_high_conditions = ['clear',  'wind', 'fog', 'cloudy', 'partly-cloudy']
_low_conditions = ['rain', 'snow', 'sleet',]
WEATHER_CONDITIONS = _high_conditions + _low_conditions

WIND_DIRECTION = {'N': 'North', 'NNE': 'North-Northeast', 'NNW': 'North-Northwest',
                  'ENE': 'East-Northeast', 'WNW': 'West-Northwest', 'E': 'East', 'W': 'West',
                  'S': 'South', 'SSE': 'South-Southeast', 'SSW': 'South-Southwest',
                  'ESE': 'East-Southeast', 'WSW': 'West-Southwest'}
