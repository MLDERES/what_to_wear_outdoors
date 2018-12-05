# -*- coding: utf-8 -*-
"""Top-level package for What to Wear Outdoors."""
import logging
import os
from cli import *
from clothing_options import *
from clothing_options_ml import *
from update_model import *
from weather_observation import *

__author__ = """Michael Dereszynski"""
__email__ = 'mlderes@hotmail.com'
__version__ = '0.1.0'

# Setup logging of debug messages to go to the file debug.log and the INFO messages
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='debug.log',
                    filemode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logging.getLogger('').addHandler(ch)

BASE_PATH = os.curdir
RUNNING_IN_NOTEBOOK = True
if RUNNING_IN_NOTEBOOK:
    base_path = 'c:/projects/what_to_wear_outdoors/'
else:
    base_path = '..'
