from unittest import TestCase

from dateutil.relativedelta import relativedelta
from nose.tools import ok_

from what_to_wear_outdoors.cli import parse_time, figure_out_date
import datetime as dt
TODAY = dt.date.today()


class TestCli(TestCase):
    def test_parse_time(self):
        ok_(parse_time("10am"), 10)
        ok_(parse_time("10Am"), 10)
        ok_(parse_time("10AM"), 10)
        ok_(parse_time("10 AM"), 10)
        ok_(parse_time("10PM"), 22)
        ok_(parse_time("10pM"), 22)

    def test_figure_out_date(self):
        ok_(figure_out_date('today'), TODAY)
        ok_(figure_out_date('tomorrow'), TODAY+relativedelta(days=1))
        ok_(figure_out_date('Sun'), dt.date(2018, 12, 23))

