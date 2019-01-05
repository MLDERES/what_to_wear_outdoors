from unittest import TestCase
from pytest import mark, fixture

from weather_observation import Location


@mark.parametrize("l_name, sep, expected", [
    ('72712', '/', '72712'),
    (' 72712 ', '/', '72712'),
    ('72712 ', '_', '72712'),
    ('Bentonville, AR', '_', 'AR_Bentonville'),
    (' Bentonville, AR', '_', 'AR_Bentonville'),
    ('Bentonville,AR', '_', 'AR_Bentonville'),
    ('Bentonville,AR', '/', 'AR/Bentonville'),
    (Location('72712'), '/', '72712'),
    (Location('Bentonville, AR'), '/', 'AR/Bentonville'),
])
def test_location_repl(l_name, sep, expected):
    loc = Location(location_name=l_name)
    if sep is None:
        assert expected == loc.repl()
    else:
        assert expected == loc.repl(sep=sep)


@mark.parametrize("l_name, expected", [
    ('72712', '72712'),
    (' 72712 ', '72712'),
    ('72712 ', '72712'),
    ('Bentonville, AR', 'Bentonville,AR'),
    (' Bentonville, AR', 'Bentonville,AR'),
    ('Bentonville,AR', 'Bentonville,AR'),
])
def test_location_name(l_name, expected):
    loc = Location(location_name=l_name)
    assert expected == loc.name
