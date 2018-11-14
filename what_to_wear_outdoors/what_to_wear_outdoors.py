# -*- coding: utf-8 -*-


"""Main module."""
from weather_observation import Weather
import datetime as dt

w = Weather()
fct = w.get_forecast(dt.datetime.now() + dt.timedelta(hours=2))
print(fct)

fct = w.get_forecast(dt.datetime.now() + dt.timedelta(hours=2), 'Sussex, WI')
print(fct)
'''
This application is meant to give you an idea of what the weather will be like during your next activity, the follow-on
example will even provide some suggestions on what to wear during that activity based on temperature, wind speed
chance of rain etc.
'''

# Explain the purpose of the program to the user
# Ask them to provide zip code or city
# Ask them when they will be going outside
#  A couple examples - Wed at 10am
#  11/3 15:00
