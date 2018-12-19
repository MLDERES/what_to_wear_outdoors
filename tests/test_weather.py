from unittest import TestCase

from what_to_wear_outdoors import Weather
import datetime as dt

NOW = dt.datetime.now
class TestWeather(TestCase):
    def test_random_forecast(self):
        print(Weather.random_forecast())
        print(Weather.random_forecast())
