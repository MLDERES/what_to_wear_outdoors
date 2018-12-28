"""Console script for what_to_wear_outdoors."""
import random
import sys
import click
import logging
import textwrap
import datetime as dt

from dateutil.relativedelta import relativedelta
from dateutil import parser
import calendar

NOW = dt.datetime.now()
TODAY = dt.date.today()

from what_to_wear_outdoors import Features
from what_to_wear_outdoors.outfit_predictors import RunningOutfitPredictor, RunningOutfitTranslator

if __name__ == '__main__' or __package__ == '':
    from weather_observation import Weather, Forecast, FctKeys
else:
    from .weather_observation import Weather, Forecast, FctKeys

# TODO: Manage Command Line Arguments
#  (-u for update model (with Excel), (-f to ask for forecast) (-d for default config) (no flags for walk-through)

'''
Here's what I want the interface to look like - 
    wtw --athlete --activity outdoor activity for advice on clothing default='run' options 'run, road, mtb' [cmd]
    [cmd] update - updates the model
            --filename using the filename specified here default='../data/what i wore running.xlsx'
          demo-mode - short-hand to execute without the prompts
            --activity outdoor activity for advice on clothing default='run' options 'run, road, mtb'
            --duration=duration
            --wind_speed
            --feel - feels like temperature
            --humidity
            --daylight - is there daylight or not options='true/false' default true
          predict - this is also shorthand, but gets the forecast
            --date - Sunday, Monday, etc or today, tomorrow default='today'
            --time - hour of the day can be 24hr time, 10:00, 10am, 10PM, 22:00, 22 default='top of the next hour'
            --loc - location, either zip or [city,state] default=72712   
'''

Colors = {'Title': 'blue', 'Description': 'cyan', 'Prompt': 'yellow', 'Error': 'red', 'Output': 'green',
          'Alternate_Output': 'cyan'}

valid_dow_value = ['today', 'tomorrow'] + list(calendar.day_abbr)

TITLE = f'#{"-" * 18} What to Wear Outdoors {"-" * 18}#'


def parse_time(time_string):
    return parser.parse(time_string).hour


def figure_out_date(weekday):
    if weekday == 'today':
        return NOW
    if weekday == 'tomorrow':
        return NOW + relativedelta(days=+1)

    return TODAY+relativedelta(weekday=list(calendar.day_abbr).index(weekday))


@click.group()
def cli():
    pass


@click.command('train_models')
# @click.argument('datapath', type=click.File(), default='data/what i wore running.xlsx')
def train_models():
    # TODO: Support using more than one model type to train with
    rop = RunningOutfitPredictor()
    rop.rebuild_models()


@click.command('demo')
@click.option('--duration', default=30, help='number of minutes you will be out')
@click.option('--windspeed', default=0, help='windspeed (mph) at time you will be out')
@click.option('--temp', default=75, help='forecast temperature for activity')
@click.option('--humidity', default=50.0, help='forecast humidity')
def demo_mode(duration, windspeed, temp, humidity):
    '''
    This mode is used to specfic the forecast rather than to look up a forecast and depend on the results.

    :type windspeed: float
    :param duration: number of minutes that the activity will last
    :param windspeed: wind speed (mph) at the time of the activity
    :param temp: forecasted (feels like) temperature (F)
    :param humidity: percent humidity
    :return: None
    '''
    hum_pct = humidity if humidity < 1 else humidity / 100
    click.secho(f'\nRecommendation for forecast\n\tTemp:{temp}'
                f'\n\tWind:{windspeed}\n\tHumidity:{hum_pct * 100}%\n\tActivity Duration:{duration}\n')

    rop = RunningOutfitPredictor()
    conditions = dict(feels_like=temp, wind_speed=windspeed, pct_humidity=humidity,
                      duration=duration, is_light=True)

    outfit = rop.predict_outfit(**conditions)
    rot = RunningOutfitTranslator()
    reply = rot.build_reply(outfit, conditions)
    [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(reply)]


@click.command('auto')
# @click.command('--dow')
# @click.command('--hour')
# @click.command('--location')
def auto_mode(dow, hour, location):
    print('got into auto_mode')
    input()
    w = Weather()
    forecast_dt = parse_time(hour)
    logging.debug(f'Forecast (calculated): {forecast_dt}')
    fct = w.get_forecast(forecast_dt, location)
    print(fct)
    rop = RunningOutfitPredictor()
    conditions = dict(feels_like=fct.feels_like, wind_speed=fct.wind_speed, pct_humidity=fct.humidity,
                      duration=45, is_light=True)
    rop = rop.predict_outfit(**conditions)
    rot = RunningOutfitTranslator()
    reply = rot.build_reply(rop.outfit_, **conditions)
    print('\n')
    [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(reply)]


@click.command('generate')
def data_generation_mode():
    """
    Ask the user about a few scenarios and generate data for those scenarios into a csv
    file that can become part of the model
    :return: None
    """
    [click.secho(line, fg=Colors['Description'])
     for line in textwrap.wrap('This mode generates random weather conditions and asks how you would dress '
                               'given a variety of weather conditions.\nLet\'s get started.\n', 60)]
    athlete_name = click.prompt(_prompt("Name"), default='Michael')
    activity_name = click.prompt(_prompt("Activity"), default='Run', type=click.Choice(['Run', 'Roadbike', 'MTB']))

    # TODO: Handle multiple sports for data generation
    outfit_predictor = RunningOutfitPredictor()
    while 1:
        # Generate random weather
        fct = Weather.random_forecast()
        # Display to the user
        print(fct)
        duration = max(20, round(random.normalvariate(45, 45)))
        distance = round((duration/random.triangular(8,15,10.5)),2)
        print(f'Duration: {duration} min, Distance: {distance} miles (pace = {round(duration/distance,2)})\n')
        # Ask what they would wear for each component of the outfit
        do_fct = click.prompt(_prompt('Do you want to do this one?'), default='y', type=click.Choice(['y', 'n', 'f']))
        if do_fct == 'f': break
        if do_fct == 'n': continue
        light_condition = click.prompt(_prompt('Would there be light?'), default='y', type=click.Choice(['y', 'n']))
        outfit = gather_data_for_forecast()
        fct.is_daylight = light_condition
        outfit_predictor.add_to_sample_data(athlete_name=athlete_name, forecast=fct, outfit=outfit, duration=duration,
                                            distance=distance, activity_date=fct.timestamp)

    outfit_predictor.write_sample_data()


def gather_data_for_forecast():
    """
    Collect all outfit components for a forecast from the user
    :param fct:
    :return:
    """
    outfit = {}
    op = RunningOutfitPredictor()
    for nm, opt in op.outfit_component_options.items():
        for x, v in enumerate(opt):
            print(f'{x}: {v}')
        outfit[nm] = opt[click.prompt(nm, type=click.IntRange(0, len(opt)), default=0)]
    return outfit


@click.command('main')
def main():
    click.clear()
    click.secho(f'#{"-" * (len(TITLE) - 2)}#', fg=Colors['Title'])
    click.secho(TITLE, fg=Colors['Title'])
    click.secho(f'#{"-" * (len(TITLE) - 2)}#', fg=Colors['Title'])
    [click.secho(line, fg=Colors['Description'])
     for line in textwrap.wrap('This application is meant to provide an idea'
                               ' of the weather conditions be during your next '
                               'outdoor activity (within the next week).\n'
                               'It will provide some suggestions on what to wear during that activity based '
                               'on temperature, wind speed chance of rain etc.\n\n', 60)]

    click.pause()
    click.clear()
    forecast_dt = prompt_for_date_time()
    while forecast_dt <= NOW:
        click.secho('\nSorry you must choose a date/time that is after the top of the next hour.\n', fg=Colors['Error'])
        forecast_dt = prompt_for_date_time()

    activity_location = click.prompt(
        click.style("Where are you going outside (zip or city,state)? ",
                    fg=Colors['Prompt']),
        default='Fayetteville, AR')

    activity_duration = click.prompt(_prompt("How long will you be out?"), default=45)

    w = Weather()
    logging.debug(f'Forecast date (calculated): {forecast_dt}')
    fct = w.get_forecast(forecast_dt, activity_location)
    print(fct)
    rop = RunningOutfitPredictor()
    conditions = {FctKeys.FEEL_TEMP: fct.feels_like, FctKeys.WIND_SPEED: fct.wind_speed,
                  FctKeys.HUMIDITY: fct.pct_humidity, Features.DURATION: activity_duration, Features.LIGHT: True}

    rop.predict_outfit(**conditions)
    rot = RunningOutfitTranslator()
    reply = rot.build_reply(rop.outfit_
                            , conditions)
    click.secho(f'\n#{"-" * 18}ML OUTPUT{"-" * 18}#', fg=Colors['Output'])
    [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(reply)]


def prompt_for_date_time():
    '''  Abstract the prompt for activity date/time in case we have to ask more than once to get it right
    :return: a representation of the date and time in UTC format
    '''
    activity_date = click.prompt(
        click.style("\n\nWhat day will you be going outside?", fg=Colors['Prompt']),
        default='today', type=click.Choice(valid_dow_value))
    activity_date = figure_out_date(activity_date)
    logging.debug(f'Activity date (determined): {activity_date}')

    activity_hour = click.prompt(_prompt("What hour of day will you be going outside? (24 hour format or HH AM/PM)?"),
                                 default=NOW.hour + 1, value_proc=parse_time)
    return activity_date.combine(activity_date.date(), dt.time(hour=activity_hour))


def _prompt(s: str):
    return click.style(s, fg=Colors['Prompt'])


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
