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


@fixture
def wth():
    return Weather()


@mark.parametrize("request_dt",
                  [
                      dt.datetime(2018, 1, 2, 8, 30),
                      dt.datetime(2017, 12, 5, 9),
                      param(NOW + dt.timedelta(days=1)),
                      None
                  ])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test_get_darksky_weather(location, request_dt):
    r = Weather.get_darksky_weather(location=location, when=request_dt)
    print(f'{r}')


@mark.parametrize("when",
                  [
                      dt.datetime(2018, 1, 2, 8, 30),
                      dt.datetime(2017, 12, 5, 9),
                      dt.datetime(2016, 12, 5, 9),
                      param(NOW + dt.timedelta(days=1)),
                      param(NOW + dt.timedelta(hours=12)),
                      param(NOW + dt.timedelta(days=12), marks=[mark.xfail])
                  ],
                  ids=['past_weather_2018', 'past_weather_2017', 'past_weather_2016', 'tomorrow',
                       'now+12 hours', '12 days ahead'])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test_get_weather(wth, location, when):
    r = wth.get_weather(location_name=location, when=when)
    print(f'{r}')


def test_get_historic_weather():
    w = Weather()
    df_hist = w.get_weather('72712', dt.datetime(year=2018, month=2, day=3, hour=7, minute=5))
    df_hist2 = w.get_weather('72712', dt.datetime(year=2018, month=2, day=3, hour=7))
    assert df_hist == df_hist2

@mark.parametrize("when",
                  [
                      param(NOW + dt.timedelta(days=1)),
                      param(NOW + dt.timedelta(hours=1)),
                      param(NOW + dt.timedelta(days=1, hours=3)),
                  ],
                  ids=['tomorrow_same_time', 'next_hour', 'now_plus_1_day_5_hours'])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test__get_weather_forecast(wth, when, location):
    o = wth._get_weather_forecast(location, when)
    print(f'{o}')

@mark.parametrize("when",
                  [
                      dt.datetime(2018, 11, 1, 8),
                      dt.datetime(2017, 12, 5, 9),
                      dt.datetime(2016, 12, 5, 9),
                  ],
                  ids=['past_weather_2018', 'past_weather_2017', 'past_weather_2016'])
@mark.parametrize("location", ['Bentonville, AR', '72712'])
def test__get_past_observation(wth, when, location):
    o = wth._get_past_observation(location, when)
    print(f'{o}')


def compare_two_series(series_a, series_b):
    c = [a == b for a, b in zip(series_a, series_b)]
    return all(c)


class TestWeather(TestCase):
    def test_random_forecast(self):
        print(Weather.random_forecast())
        print(Weather.random_forecast())

    def test__get_random_weather_condition(self):
        print(Weather._get_random_weather_condition_weights())
