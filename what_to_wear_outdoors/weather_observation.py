import random
from numpy.random import choice
import requests
import dotenv
import os
import re
import logging
import json

__all__ = ['Forecast', 'Weather']

# TODO: Put this back so that we aren't shipping the API key with the code
dotenv.load_dotenv()
WU_API_KEY = os.getenv("WU_API_KEY")
if WU_API_KEY is None:
    raise EnvironmentError(
        "The environment variable WU_API_KEY is required to run this program.")

# So lets just key our observations off of mon_day_year
#  We'll store pretty date for whatever reason
FORECAST_KEY = 'hourly_forecast'
FCAST_TIME_KEY = 'FCTTIME'
DATE_KEY = 'UTCDATE'
DATE_HOUR = 'hour'
DATE_MONTH = 'mon'
DATE_DAY = 'mday'
DATE_YEAR = 'year'
DAY_OF_WEEK = 'weekday_name'
CIVIL = 'civil'

CONDITION_KEY = 'condition'  # this would be the technical description
FANCY_COND_KEY = 'wx'  # this is the human readable condition
FEELS_LIKE_KEY = 'feelslike'  # then there is a dict of 'english' or 'metric'
METRIC_KEY = 'metric'
ENG_KEY = 'english'
HEAT_IDX_KEY = 'heatindex'
UV_INDEX_KEY = 'uvi'
TEMP_KEY = 'temp'
WIND_DIR_KEY = 'wdir'  # but then we need the 'dir' or 'degrees' key to get it
WIND_SPEED_KEY = 'wspd'  # also read dictionary of 'english' or 'metric'
RAIN_CHANCE_KEY = 'pop'  # probability of precipitation
NULL_VALUE = -999
HUMIDITY_KEY = 'humidity'
WIND_DIRECTION = {'N': 'North', 'NNE': 'North-Northeast', 'NNW': 'North-Northwest',
                  'ENE': 'East-Northeast', 'WNW': 'West-Northwest', 'E': 'East', 'W': 'West',
                  'S': 'South', 'SSE': 'South-Southeast', 'SSW': 'South-Southwest',
                  'ESE': 'East-Southeast', 'WSW': 'West-Southwest'}


class JsonDictionary(dict):

    def __missing__(self, key):
        return None

    def read_int(self, key):
        return None if self[key] is None or int(self[key]) <= NULL_VALUE else int(self[key])

    def read_float(self, key):
        return None if float(self[key]) <= NULL_VALUE else float(self[key])


class Forecast:
    WIND_DIRECTIONS = ['N', 'NNE', '']

    def __init__(self, location="", condition="", condition_human="", feels_like_f=0, heat_index_f=0, temp_f=0,
                 wind_dir='', wind_speed=0, precip_chance=0, tod=0, month_day=1, mth=1, dow='Unknown',
                 civil_time='00:00 AM', humidity=0, is_daylight=True):
        super(Forecast, self).__init__()
        self.location = location
        self.condition = condition
        self.condition_human = condition_human
        self.feels_like_f = feels_like_f
        self.heat_index_f = heat_index_f
        self.temp_f = temp_f
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
        self.precip_chance = precip_chance
        self.tod = tod
        self.month_day = month_day
        self.mth = mth
        self.dow = dow
        self.civil_time = civil_time
        self.humidity = humidity
        self.is_daylight = is_daylight

    def __str__(self):
        return f'Forecast for {self.location} ' \
            f'on {self.dow} (Month-Day): {self.mth}-{self.month_day} ' \
            f'at {self.civil_time}. ' \
            f'\n\tTemperature (feels like): {self.feels_like_f} °F' \
            f'\n\tWind {self.wind_speed} mph from {self.wind_dir}' \
            f'\n\tConditions: {self.condition_human} ' \
            f'\n\tChance of precipitation: {self.precip_chance} %' \
            f'\n\tHumidity: {self.humidity}'

    @staticmethod
    def get_fct_key(d=0, m=0, h=0):
        return "{}_{}_{}".format(m, d, h)

    @staticmethod
    def _read_int(i):
        return None if int(i) <= NULL_VALUE else int(i)

    def _read_float(self, f):
        return None if float(f) <= NULL_VALUE else float(f)

    def from_dict(self, dct):
        fcast = Forecast()
        time_dct = JsonDictionary(dct[FCAST_TIME_KEY])
        fcast.tod = time_dct.read_int(DATE_HOUR)
        fcast.mth = time_dct.read_int(DATE_MONTH)
        fcast.dow = time_dct[DAY_OF_WEEK]
        fcast.civil_time = time_dct[CIVIL]
        fcast.month_day = time_dct.read_int(DATE_DAY)
        fcast.condition = dct[CONDITION_KEY]
        fcast.condition_human = dct[FANCY_COND_KEY]
        fcast.feels_like_f = self._read_float(dct[FEELS_LIKE_KEY][ENG_KEY])
        fcast.heat_index_f = self._read_float(dct[HEAT_IDX_KEY][ENG_KEY])
        fcast.temp_f = self._read_float(dct[TEMP_KEY][ENG_KEY])
        fcast.wind_dir = dct[WIND_DIR_KEY]['dir']
        fcast.wind_speed = self._read_float(dct[WIND_SPEED_KEY][ENG_KEY])
        fcast.precip_chance = self._read_float(dct[RAIN_CHANCE_KEY])
        fcast.humidity = self._read_float(dct[HUMIDITY_KEY])

        return fcast


class Weather:

    def __init__(self):
        pass

    @staticmethod
    def random_forecast() -> Forecast:
        """ Get a random forecast.

            Used for building up the dataset

        :return: a Forecast
        """
        f = Forecast()
        f.humidity = round(random.normalvariate(55, 20))
        f.temp_f = f.feels_like_f = round(random.normalvariate(55, 20))
        f.tod = random.randrange(5, 21)
        pchance = random.randrange(0, 100, step=10)
        f.precip_chance = 0 if pchance < 20 else (100 if pchance > 80 else pchance)
        f.wind_speed = min(0, round(random.normalvariate(8, 8)))
        f.wind_dir = random.choice(list(WIND_DIRECTION.keys()))
        f.condition_human = random.choice(WEATHER_CONDITIONS)
        return f

    def _build_forecasts(self, dct, location=''):
        forecasts = {}
        if FORECAST_KEY in dct:
            for f in dct[FORECAST_KEY]:
                time_dct = JsonDictionary(f[FCAST_TIME_KEY])
                f_key = Forecast.get_fct_key(d=time_dct.read_int(DATE_DAY),
                                             m=time_dct.read_int(DATE_MONTH),
                                             h=time_dct.read_int(DATE_HOUR))
                fct = Forecast()
                forecasts[f_key] = fct.from_dict(f)
                forecasts[f_key].location = location
        return forecasts

    '''
    This function gets an hourly forecast for the next 10 days.
    '''

    def get_forecast(self, dt, location: str = '72712', dbg: bool = False) -> Forecast:
        """
        Get the forecast for a given location and a given day/time.
        :param dt: the day and time for when the forecast should be queried
        :type location: string
        :param location: location should be either a zip code or a city, state abbreviation 
        :param bool param_name: dbg: if True, read the forecast from a file rather than querying the web service
        :return: a Forecast object
        """"""

        """
        # dt should be the date and the time
        logging.debug('get_weather location = ' + location)
        logging.debug('date time = {}'.format(dt))

        fct_key = Forecast.get_fct_key(dt.day, dt.month, dt.hour)
        if _All_Forecasts_Location[location] is None or _All_Forecasts_Location[location][fct_key] is None:
            weather_request = Weather._build_weather_request(
                location)  # +'/geolookup/conditions/hourly/q/'+city+'.json'
            resp = requests.get(weather_request)
            if resp.status_code == 200:
                wu_response = resp.json()
                if dbg:
                    with open('sample_forecast.json', 'w') as fp:
                        json.dump(wu_response, fp)
                _All_Forecasts_Location[location] = self._build_forecasts(wu_response, location)
            else:
                print(f'Danger Will Robinson we got a bad response from WU {resp.status_code}')

        return _All_Forecasts_Location[location][fct_key]

    @staticmethod
    def _build_location_query(location):
        query_loc = location
        # This expression test for a Zip code, a City, State or just a city
        expression = r'(^\d{5}$)|(^[\w\s]+),\s*(\w{2}$)|(^[\w\s]+)'
        mo = re.match(expression, str(location))
        if mo and mo.group(2) is not None:
            # if we have matched City, State then we need to build the query as ST/City.json
            # otherwise we can just use City or Zip + .json
            query_loc = f'{mo.group(3)}/{mo.group(2)}'
        return query_loc

    @staticmethod
    def _build_weather_request(location):
        # http://api.wunderground.com/api/7d65568686ff9c25/features/settings/q/query.format
        # Features = alerts/almanac/astromony/conditions/forecast/hourly/hourly10day etc.
        # settings(optional) = lang, pws(personal weather stations):0 or 1
        # query = location (ST/City, zipcode,Country/City, or lat,long)
        # format = json or xml
        request = f'http://api.wunderground.com/api/{WU_API_KEY}/hourly10day/q/' \
            f'{Weather._build_location_query(location)}.json'
        return request


# Keep track of all the forecasts we have gotten by location
_All_Forecasts_Location = JsonDictionary()
WEATHER_CONDITIONS = ['Clear', 'Heavy Fog', 'Heavy Fog Patches', 'Heavy Freezing Drizzle', 'Heavy Freezing Fog',
                      'Heavy Freezing Rain', 'Heavy Hail', 'Heavy Hail Showers', 'Heavy Haze', 'Heavy Mist',
                      'Heavy Rain', 'Heavy Rain Mist',
                      'Heavy Rain Showers', 'Heavy Small Hail Showers', 'Heavy Snow', 'Heavy Snow Blowing Snow Mist',
                      'Heavy Snow Grains', 'Heavy Snow Showers',
                      'Heavy Spray', 'Heavy Thunderstorm', 'Heavy Thunderstorms and Ice Pellets',
                      'Heavy Thunderstorms and Rain',
                      'Heavy Thunderstorms and Snow', 'Heavy Thunderstorms with Hail',
                      'Heavy Thunderstorms with Small Hail', 'Heavy Drizzel',
                      'Light Fog', 'Light Fog Patches', 'Light Freezing Drizzle', 'Light Freezing Fog',
                      'Light Freezing Rain', 'Light Hail',
                      'Light Hail Showers', 'Light Haze', 'Light Mist', 'Light Rain', 'Light Rain Mist',
                      'Light Rain Showers', 'Light Sandstorm',
                      'Light Small Hail Showers', 'Light Snow', 'Light Snow Blowing Snow Mist', 'Light Snow Grains',
                      'Light Snow Showers',
                      'Light Spray', 'Light Thunderstorm', 'Light Thunderstorms and Ice Pellets',
                      'Light Thunderstorms and Rain',
                      'Light Thunderstorms and Snow', 'Light Thunderstorms with Hail',
                      'Light Thunderstorms with Small Hail',
                      'Light Drizzel', 'Mostly Cloudy', 'Overcast', 'Partial Fog', 'Partly Cloudy', 'Patches of Fog',
                      'Scattered Clouds',
                      'Shallow Fog', 'Small Hail', 'Squalls', 'Unknown', 'Unknown Precipitation']
