
"""Console script for what_to_wear_outdoors."""
import datetime as dt
import sys
import re
import click
from weather_observation import Weather, Forecast
from clothing_options import Running
import logging
import textwrap

__all__ = ['main']

Colors = {'Title': 'blue', 'Description': 'cyan', 'Prompt': 'yellow', 'Error': 'red', 'Output': 'green'}

days_of_the_week = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
NOW = dt.datetime.now()
TITLE = f'#{"-"*18} What to Wear Outdoors {"-"*18}#'

''' Abstract the prompt for activity date/time in case we have to ask more than once to get it right
'''


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

    # @click.command()


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
    click.secho(f'#{"-"*(len(TITLE)-2)}#', fg=Colors['Title'])
    click.secho(TITLE, fg=Colors['Title'])
    click.secho(f'#{"-"*(len(TITLE)-2)}#', fg=Colors['Title'])
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
    [click.secho(li, fg=Colors['Output']) for li in textwrap.wrap(running.build_reply(fct))]


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
    if (weekday == 'today'): return NOW
    if (weekday == 'tomorrow'): return NOW + dt.timedelta(days=1)
    dow_target = days_of_the_week.index(weekday.lower()[:3])
    dow_today = dt.datetime.weekday(NOW)
    days_ahead = dow_target - dow_today
    if (dow_target < dow_today):
        days_ahead = + 7
    return dt.datetime.today() + dt.timedelta(days=days_ahead)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
