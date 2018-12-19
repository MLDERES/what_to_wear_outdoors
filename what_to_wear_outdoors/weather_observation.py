import calendar
import random
import requests
import dotenv
import os
import re
import logging
import json
import datetime as dt
import dateutil as dtu

NOW = dt.datetime.now()

__all__ = ['Forecast', 'Weather']

# TODO: Put this back so that we aren't shipping the API key with the code
dotenv.load_dotenv()
WU_API_KEY = os.getenv("WU_API_KEY")
if WU_API_KEY is None:
    raise EnvironmentError(
        "The environment variable WU_API_KEY is required to run this program.")

# So lets just key our observations off of mon_day_year
#  We'll store pretty date for whatever reason
DATE_KEY = 'UTCDATE'
DATE_HOUR = 'hour'
DATE_MONTH = 'mon'
DATE_DAY = 'mday'
DATE_YEAR = 'year'
DAY_OF_WEEK = 'weekday_name'
FORECAST_KEY = 'hourly_forecast'
FCAST_TIME_KEY = 'FCTTIME'
NULL_VALUE = -999


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


class JsonDictionary(dict):

    def __missing__(self, key):
        return None

    def read_int(self, key):
        return None if self[key] is None or int(self[key]) <= NULL_VALUE else int(self[key])

    def read_float(self, key):
        return None if float(self[key]) <= NULL_VALUE else float(self[key])


class Forecast:

    def __init__(self):
        super(Forecast, self).__init__()
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
        self.is_daylight = True
        self.pct_humidity = 50
        self.timestamp = dt.datetime(year=NOW.year, month=self.mth, day=self.month_day, hour=self.tod)

    def __str__(self):
        return f'Forecast for {self.location} ' \
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
        fcast = Forecast()
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
        pass

    @staticmethod
    def random_forecast() -> Forecast:
        """ Get a random forecast.

            Used for building up the dataset

        :return: a Forecast
        """
        f = Forecast()
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
                                     weights=([60] * len(_high_conditions)) + ([15] * len(_med_conditions)) +
                                             ([5] * len(_low_conditions)) + ([1] * len(_unlikely_conditions)))
        return f

    @staticmethod
    def _build_forecasts(dct, location=''):
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

_high_conditions = ['Clear', 'Overcast', 'Partial Fog', 'Partly Cloudy', 'Patches of Fog',
                    'Scattered Clouds', 'Mostly Cloudy', 'Light Rain', ]
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
WEATHER_CONDITIONS = _high_conditions + _med_conditions + _low_conditions + _unlikely_conditions

WIND_DIRECTION = {'N': 'North', 'NNE': 'North-Northeast', 'NNW': 'North-Northwest',
                  'ENE': 'East-Northeast', 'WNW': 'West-Northwest', 'E': 'East', 'W': 'West',
                  'S': 'South', 'SSE': 'South-Southeast', 'SSW': 'South-Southwest',
                  'ESE': 'East-Southeast', 'WSW': 'West-Southwest'}
