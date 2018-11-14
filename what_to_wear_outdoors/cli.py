# -*- coding: utf-8 -*-

"""Console script for what_to_wear_outdoors."""
import sys
import click
import datetime as dt
days_of_the_week = ['monday','tuesday','wednesday','thursday','friday','saturday', 'sunday']

# @click.command()
# @click.option('--date', default=dt.datetime.now().date(), help='date that you will be going outside', prompt=True)
# @click.option('--day', default=dt.datetime.now().weekday())
# @click.option('--hour', default=dt.datetime.now().hour, help='hour that you will be going outside', prompt=True,
#               type=click.IntRange(0-23))
# @click.option('--activity', type=click.Choice(['run','bike']), prompt=True,
#               help='the type of activity you will be doing outside')
# @click.option('--zip', help='the zipcode of the location for your activity',
#             prompt='The zipcode for the location where you''ll be going out?', type=click.IntRange(0-99999))
#def main(date, hour, zip, activity,day):
def main():
    """Console script for what_to_wear_outdoors."""
    click.echo('This application is meant to give you an idea of what the weather will be like '
                'during your next activity.')

    click.echo('It will provide some suggestions on what to wear during that activity based on'
                'temperature, wind speed chance of rain etc.')
    #click.echo(date)
    click.prompt("What day will you be going outside? (Sunday, Monday, etc.)?", 'today')
    click.prompt("What time of day will you be going outside? (24 hour format or HH AM/PM)?", dt.datetime.now().hour)
    #click.echo(hour)
    #click.echo(activity)
    #click.echo(zip)
    #click.echo(day)
    #click.echo()

    return 0

def parse_time(time_string):
    import re
    re.search(r'([0-2]{0,1}[0-9]{0,1})(:{0,1})([0-5][0-9]){0,1}\s*([AaPp][mM]){0,1}')
    # This should match 2300, 23:00, 1:00pm, 1:00PM, 1pm, etc.
    # Group 1 is the hour, Group 4 is AM/PM if necessary
    
def figure_out_date(weekday):
    if(weekday == 'today'): return dt.datetime.today()
    dow_target = days_of_the_week.index(weekday.lower())
    dow_today = dt.datetime.weekday(dt.datetime.now())
    days_ahead = dow_target-dow_today
    if (dow_target < dow_today):
        days_ahead =+ 7
    return dt.datetime.today() + dt.timedelta(days=days_ahead)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
