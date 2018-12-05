#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `what_to_wear_outdoors` package."""

import unittest
from click.testing import CliRunner
from what_to_wear_outdoors import cli
from what_to_wear_outdoors.weather_observation import Weather
from what_to_wear_outdoors.clothing_options_ml import Running


class Test_what_to_wear_outdoors(unittest.TestCase):
    """Tests for `what_to_wear_outdoors` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    # Test to ensure that the correct query string is being created for weather query locations
    def test_build_query_string(self):
        assert (Weather._build_location_query(72712) == 72712)
        assert (Weather._build_location_query('Fayetteville, AR') == 'AR/Fayetteville')
        assert (Weather._build_location_query('Fayetteville,AR') == 'AR/Fayetteville')
        assert (Weather._build_location_query('Bentonville') == 'Bentonville')
        assert (Weather._build_location_query('72712') == '72712')

    def test_parse_time(self):
        assert (cli.parse_time('10:00 PM') == 22)
        assert (cli.parse_time('10:00 am') == 10)
        assert (cli.parse_time('10:05 AM') == 10)
        assert (cli.parse_time('10:00 pm') == 22)
        assert (cli.parse_time('10:00 pM') == 22)
        assert (cli.parse_time('10:00 Am') == 10)
        assert (cli.parse_time('10') == 10)
        assert (cli.parse_time('23') == 23)
        assert (cli.parse_time('5') == 5)
        assert (cli.parse_time('830pm') == 20)
        assert (cli.parse_time('830 aM') == 8)
        assert (cli.parse_time('830') == 8)
        assert (cli.parse_time('2300') == 23)
        assert (cli.parse_time('1000') == 10)
        assert (cli.parse_time('1005') == 10)
        assert (cli.parse_time('10AM') == 10)
        assert (cli.parse_time('10PM') == 22)
        assert (cli.parse_time('10 AM') == 10)
        assert (cli.parse_time('10 PM') == 22)
        assert (cli.parse_time('10 am') == 10)
        assert (cli.parse_time('10 pm') == 22)

    def test_figure_out_date(self):
        pass

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'what_to_wear_outdoors.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_predict_clothing(self):
        r = Running()
        assert (r.pred_clothing(10, 55, 40, 'shorts', light=True))
