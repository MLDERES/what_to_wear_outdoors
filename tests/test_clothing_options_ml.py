#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from what_to_wear_outdoors.clothing_options_ml import Running
from what_to_wear_outdoors.weather_observation import Weather


class Test_clothing_options_ml(unittest.TestCase):
    """Tests for `what_to_wear_outdoors` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    # Test to ensure that the correct query string is being created for weather query locations
    def test_predict_clothing(self):
        r = Running()
        assert (r.pred_clothing(10, 55, 40, 'shorts', light=True))
