import math
from datetime import datetime, timedelta
import pandas as pd

def course_week(date_string, course_start_date):
    """
    Find the course week for the event time
    """
    date = None
    if isinstance(date_string, str):
        if len(date_string) > 18:
            date_string = date_string[0:18]
            date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
        else:
            date_string = date_string[0:8]
            date = datetime.strptime(date_string, "%Y%m%d")
    else:
        date = date_string
    
    date = datetime.combine(date, datetime.min.time())
    course_start_date = datetime.combine(course_start_date, datetime.min.time())

    course_week = math.ceil((date - course_start_date).days / 7)
    if course_week == 0:
        course_week = 1
    return course_week

def two_weeks_ago():
    recent = datetime.today() - timedelta(days=14)
    return recent.strftime('%Y-%m-%d')
