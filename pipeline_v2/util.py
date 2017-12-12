import math
from datetime import datetime

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
    elif isinstance(date_string, datetime):
        date = date_string

    course_week = math.ceil((date - course_start_date).days / 7)
    if course_week == 0:
        course_week = 1
    return course_week

def get_course_dates(course_dates_df):
    """
    Get the start and end dates for the course
    """

    def get_datetime_col(col_name):
        """
        Get column as a datetime object
        """
        return datetime.strptime(course_dates_df[col_name][0], '%Y-%m-%d')

    course_start_date = get_datetime_col('CourseRunStartDate')
    course_end_date = get_datetime_col('CourseRunEndDate')
    return (course_start_date, course_end_date)