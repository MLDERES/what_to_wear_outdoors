"""Console script for what_to_wear_outdoors."""
import datetime as dt
import sys
import re
import click
import logging
import textwrap

from what_to_wear_outdoors.clothing_options import Running2

if __name__ == '__main__':
    from weather_observation import Weather, Forecast
    from clothing_options_ml import Running
    from train_model import train
else:
    from .weather_observation import Weather, Forecast
    from .clothing_options_ml import Running
    from .train_model import train

# TODO: Manage Command Line Arguments
#  (-u for update model (with Excel), (-f to ask for forecast) (-d for default config) (no flags for walk-through)
Colors = {'Title': 'blue', 'Description': 'cyan', 'Prompt': 'yellow', 'Error': 'red', 'Output': 'green',
          'Alternate_Output': 'cyan'}

days_of_the_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
valid_dow_value = ['today', 'tomorrow'] + days_of_the_week

NOW = dt.datetime.now()
TITLE = f'#{"-" * 18} What to Wear Outdoors {"-" * 18}#'


def parse_time(time_string):
    m = re.search(r'([0-2]{0,1}[0-9]{0,1})(:{0,1})([0-5][0-9]){0,1}\s*([AaPp][mM]){0,1}', time_string)
    # This should match 2300, 23:00, 1:00pm, 1:00PM, 1pm, etc.
    # Group 1 is the hour, Group 4 is AM/PM if necessary
    hour = NOW.hour
    if m is not None:
        hour = int(m.group(1))
        if m.group(4) is not None and m.group(4).lower() == 'pm':
            hour += 12
    return hour


def figure_out_date(weekday):
    if weekday == 'today':
        return NOW
    if weekday == 'tomorrow':
        return NOW + dt.timedelta(days=1)
    dow_target = days_of_the_week.index(weekday.lower()[:3])
    dow_today = dt.datetime.weekday(NOW)
    days_ahead = dow_target - dow_today
    if (dow_target < dow_today):
        days_ahead = + 7
    return dt.datetime.today() + dt.timedelta(days=days_ahead)


''' Abstract the prompt for activity date/time in case we have to ask more than once to get it right
'''


@click.group()
def cli():
    pass


@click.command('train_models')
# @click.argument('datapath', type=click.File(), default='data/what i wore running.xlsx')
def train_models():
    train()


@click.command('demo')
# @click.option('--duration', default=30, help='number of minutes you will be out')
# @click.option('--windspeed', default=0, help='windspeed (mph) at time you will be out')
# @click.option('--temp', default=70, help='forecast temperature for activity')
# @click.option('--humidity', default=50.0, help='forecast humidity')
def demo_mode(duration, windspeed, temp, humidity):
    print('got into demo mode')
#     input('press a key to continue')
#     r = Running()
#     prediction = r.pred_clothing(duration=duration, wind_speed=windspeed,
#                                  feel=temp, hum=humidity, item=OUTFIT_COMPONENTS, light=True)
#     [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(prediction)]


@click.command('auto')
# @click.command('--day', type=click.Choice(valid_dow_value), prompt=True)
# @click.command('--hour', value_proc=parse_time)
# @click.command('--location')
def auto_mode(day, hour, location):
    print('got into auto_mode')
    # w = Weather()
    # logging.debug(f'Forecast (calculated): {forecast_dt}')
    # fct = w.get_forecast(forecast_dt, activity_location)
    # print(fct)
    # running = Running()
    # print('\n')
    # [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(running.build_reply(fct))]


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


# @click.option('--date', default=dt.datetime.now().date(), help='date that you will be going outside', prompt=True)
# @click.option('--day', default=dt.datetime.now().weekday())
# @click.option('--hour', default=dt.datetime.now().hour, help='hour that you will be going outside', prompt=True,
#               type=click.IntRange(0-23))
# @click.option('--activity', type=click.Choice(['run','bike']), prompt=True,
#               help='the type of activity you will be doing outside')
# @click.option('--zip', help='the zipcode of the location for your activity',
#             prompt='The zipcode for the location where you''ll be going out?', type=click.IntRange(0-99999))
# def main(date, hour, zip, activity,day):
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
    while (forecast_dt <= NOW):
        click.secho('\nSorry you must choose a date/time that is after the top of the next hour.\n', fg=Colors['Error'])
        forecast_dt = prompt_for_date_time()

    activity_location = click.prompt(
        click.style("Where are you going outside (zip or city,state)? ",
                    fg=Colors['Prompt']),
        default='Fayetteville, AR')

    w = Weather()
    logging.debug(f'Forecast (calculated): {forecast_dt}')
    fct = w.get_forecast(forecast_dt, activity_location)
    print(fct)
    running = Running()
    print('\n')
    click.secho(f'#{"-" * 18}ML OUTPUT{"-" * 18}#',fg=Colors['Output'])
    [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(running.build_reply(fct))]
    running2 = Running2()
    click.secho(f'\n\n#{"-" * 18}STANDARD OUTPUT{"-" * 18}#', fg=Colors['Alternate_Output'])
    [click.secho(li, fg=Colors['Alternate_Output']) for li in textwrap.wrap(running2.build_reply(fct))]

def prompt_for_date_time():
    activity_date = click.prompt(
        click.style("\n\nWhat day will you be going outside? (Sunday, Monday, etc. "
                    "also today, tomorrow work as well.) ", fg=Colors['Prompt']),
        default='today', value_proc=figure_out_date)
    if (type(activity_date) == str):
        activity_date = figure_out_date(activity_date)
    logging.debug(f'Activity date (determined): {activity_date}')

    activity_hour = click.prompt(click.style("What hour of day will you be going outside? "
                                             "(24 hour format or HH AM/PM)?", fg=Colors['Prompt']),
                                 default=NOW.hour + 1, value_proc=parse_time)
    return activity_date.combine(activity_date.date(), dt.time(hour=activity_hour))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
