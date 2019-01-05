from pathlib import Path
from unittest import TestCase
import json
from pytest import mark, param, fixture
from what_to_wear_outdoors import Weather
import datetime as dt

NOW = dt.datetime.now()
TODAY = NOW.date()


@fixture
def historical_json():
    historical_json = Path('C:/projects/what_to_wear_outdoors/tests/sample_historic_weather.json')
    with open(historical_json) as json_data:
        j = json.load(json_data)
    return j


@fixture
def forecast_json():
    fct_json = Path('C:/projects/what_to_wear_outdoors/tests/sample_forecast.json')
    with open(fct_json) as json_data:
        j = json.load(json_data)
    return j


@mark.parametrize("request_dt",
                  [
                      dt.datetime(2018, 1, 2, 8, 30),
                      dt.datetime(2017, 12, 5, 9),
                      param(NOW + dt.timedelta(days=1), marks=[mark.xfail])
                  ])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test__build_historical_request(request_dt, location):
    r = Weather._build_historical_request(location=location, d=request_dt)
    print(f'{r}')


@mark.parametrize("when",
                  [
                      dt.datetime(2018, 1, 2, 8, 30),
                      dt.datetime(2017, 12, 5, 9),
                      param(NOW + dt.timedelta(days=1)),
                      param(NOW + dt.timedelta(days=12), marks=[mark.xfail])
                  ])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test_get_weather(location, when):
    w = Weather()
    r = w.get_weather(location_name=location, when=when)
    print(f'{r}')
    assert 1


def test_historic_forecast_same_cols(forecast_json, historical_json):
    df_fct = Weather._build_forecast_df(forecast_json, '72712')
    df_historic = Weather._build_historic_df(historical_json, '72712')
    print(f'\nColumns in historic and not in fct:\n:'
          f'{[x for x in df_historic.columns if x not in df_fct.columns]}')
    print(f'\nColumns in fct and not in historic:\n:'
          f'{[x for x in df_fct.columns if x not in df_historic.columns]}')


def test_get_historic_weather():
    w = Weather()
    df_hist = w.get_weather('72712', dt.datetime(year=2018, month=2, day=3, hour=7, minute=5))
    df_hist2 = w.get_weather('72712', dt.datetime(year=2018, month=2, day=3, hour=7))
    assert compare_two_series(df_hist, df_hist2)



def compare_two_series(series_a, series_b):
    c = [a == b for a, b in zip(series_a, series_b)]
    return all(c)


class TestWeather(TestCase):
    def test_random_forecast(self):
        print(Weather.random_forecast())
        print(Weather.random_forecast())

    def test__get_random_weather_condition(self):
        print(Weather._get_random_weather_condition_weights())
