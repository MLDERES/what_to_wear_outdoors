from unittest import TestCase

from what_to_wear_outdoors import Weather
import datetime as dt

NOW = dt.datetime.now


class TestWeather(TestCase):
    def test_random_forecast(self):
        print(Weather.random_forecast())
        print(Weather.random_forecast())


    def test__get_random_weather_condition(self):
        print(Weather._get_random_weather_condition_weights())


