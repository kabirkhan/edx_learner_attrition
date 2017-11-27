"""
Utility functions for the pipeline
"""
import os
import errno
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


def save_df_to_csv(dataframe, name, course_id):
    """
    Save a dataframe to a csv file in the data/{course_id}/ directory
    """
    path = '{}/{}/{}.csv'.format(get_data_path(), course_id, name)
    create_directory_safe(path)
    dataframe.to_csv(path, index=False)


def get_data_path():
    """
    Return the path to the data folder
    """
    return '{}/data'.format(os.path.dirname(os.path.realpath(sys.argv[0])))


def create_directory_safe(path):
    """
    Safely create a directory
    """
    try:
        if '.' in path:
            directory = os.path.dirname(path)
        else:
            directory = path
        os.makedirs(directory)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
