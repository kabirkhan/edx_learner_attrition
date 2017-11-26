"""
Utility functions for the pipeline
"""
import os
import sys
import math
from datetime import datetime


def course_week(date_string, course_start_date):
    """
    Find the course week for the event time
    """
    if isinstance(date_string, str):
        if len(date_string) > 18:
            date_string = date_string[0:18]
            date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
        else:
            date_string = date_string[0:8]
            date = datetime.strptime(date_string, "%Y%m%d")
    elif isinstance(date_string, datetime):
        date = date_string

    course_week = math.ceil((date - course_start_date).days / 7)
    if course_week == 0:
        course_week = 1
    return course_week


def get_data_path():
    return '{}/data'.format(os.path.dirname(os.path.realpath(sys.argv[0])))
