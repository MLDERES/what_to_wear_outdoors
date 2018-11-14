import json
import logging
import requests
import dotenv
import os
import re

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


class JsonDictionary(dict):

    def __missing__(self, key):
        return None

    def read_int(self, key):
        return None if self[key] is None or int(self[key]) <= NULL_VALUE else int(self[key])

    def read_float(self, key):
        return None if float(self[key]) <= NULL_VALUE else float(self[key])


class Forecast():
    def __init__(self):
        super(Forecast, self).__init__()
        self.location = ""
        self.condition = ""
        self.condition_human = ""
        self.feels_like_f = 0
        self.heat_index_f = 0
        self.temp_f = 0
        self.wind_dir = ''
        self.wind_speed = 0
        self.precip_chance = 0
        self.tod = 0
        self.month_day = 1
        self.mth = 1

    def __str__(self):
        return f'Forecast for {self.location} ' \
               f'on Month-Day: {self.mth}-{self.month_day} ' \
               f'@ Hour: {self.tod}. Temp(feels like f):{self.feels_like_f} ' \
               f'Condition:{self.condition} ' \
               f'Precip Percentage: {self.precip_chance}'

    @staticmethod
    def get_fct_key(d=0, m=0, h=0):
        return "{}_{}_{}".format(h, d, m)

    def _read_int(self, i):
        return None if int(i) <= NULL_VALUE else int(i)

    def _read_float(self, f):
        return None if float(f) <= NULL_VALUE else float(f)

    def from_dict(self, dct):
        fcast = Forecast()
        time_dct = JsonDictionary(dct[FCAST_TIME_KEY])
        fcast.tod = time_dct.read_int(DATE_HOUR)
        fcast.mth = time_dct.read_int(DATE_MONTH)
        fcast.month_day = time_dct.read_int(DATE_DAY)
        fcast.condition = dct[CONDITION_KEY]
        fcast.condition_human = dct[FANCY_COND_KEY]
        fcast.feels_like_f = self._read_float(dct[FEELS_LIKE_KEY][ENG_KEY])
        fcast.heat_index_f = self._read_float(dct[HEAT_IDX_KEY][ENG_KEY])
        fcast.temp_f = self._read_float(dct[TEMP_KEY][ENG_KEY])
        fcast.wind_dir = dct[WIND_DIR_KEY]['dir']
        fcast.wind_speed = self._read_float(dct[WIND_SPEED_KEY][ENG_KEY])
        fcast.precip_chance = self._read_float(dct[RAIN_CHANCE_KEY])
        return fcast


class Weather:

    def __init__(self):
        pass

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
    def get_forecast(self, dt, location='72712', dbg=False):
        # dt should be the date and the time
        logging.debug('get_weather location = ' + location)
        logging.debug('date time = {}'.format(dt))

        fct_key = Forecast.get_fct_key(dt.day, dt.month, dt.hour)
        if (_All_Forecasts_Location[location] is None or _All_Forecasts_Location[location][fct_key] is None):
            weather_request = Weather._build_weather_request(
                location)  # +'/geolookup/conditions/hourly/q/'+city+'.json'
            resp = requests.get(weather_request)
            if (resp.status_code == 200):
                wu_response = resp.json()
                _All_Forecasts_Location[location] = self._build_forecasts(wu_response, location)
            else:
                print('Danger Will Robinson we got a bad response from WU {resp.status_code}')

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
