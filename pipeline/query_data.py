import os
import sys
import logging
import pymssql
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
from collections import Counter
from configparser import SafeConfigParser
from pipeline.util import *

parser = SafeConfigParser()
parser.read('setup.cfg')

LONG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
SHORT_DATE_FORMAT = "%Y%m%d"

LOG_CONFIG = {'root':{'handlers':('console', 'file'), 'level':'DEBUG'}}

FORMAT = '%(name)s:   %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOG = logging.getLogger(__name__)

def query_data(course_id, from_checkpoint=False):

    if from_checkpoint:
        def get_data_from_file(name):            
            return pd.read_csv("{}/{}/{}.csv".format(get_data_path(), course_id, name))
        events = get_data_from_file('events')
        forums = get_data_from_file('forums')
        course_starts = get_data_from_file('course_starts')
        course_completions = get_data_from_file('course_completions')
        course_dates = get_data_from_file('course_dates')
        course_start_date, course_end_date = get_course_dates(course_dates)
    else:
        DB_USER = os.environ['DB_USER']
        DB_PASS = os.environ['DB_PASSWORD']
        DB_SERVER = os.environ['DB_SERVER']
        CONN = pymssql.connect(DB_SERVER, DB_USER, DB_PASS)
        ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))

        EVENTS_QUERY = """
        SELECT
            UserId, EventType, EventSource, CourseId,
            EventGrade, EventAttempts, EventMaxGrade, 
            EventSub_Correct, EventTime
        FROM [EdxStaging].[edx].[Edx_DailyEvents]
        WHERE (Host = 'courses.edx.org' and CourseId = '{}')
        AND UserId IS NOT NULL
        """.format(course_id)

        FORUMS_QUERY = """
        SELECT [Title]
            ,[CommentText]
            ,[AuthorId]            
            ,[VotesUpCount]
            ,[VotesDownCount]
            ,[VotesCount]
            ,[VotesPoint]              
            ,[CommentCount]      
            ,[ParentId]
            ,[CommentThreadId]
            ,[CourseId]
            ,[TextType]       
            ,[UpdateTimestamp]
        FROM [EdxStaging].[edx].[Edx_Forum]
        WHERE CourseId = '{}'
        """.format(course_id)

        COURSE_START_QUERY = """
        SELECT U.UserId as user_id
            ,CS.[DateKey] as date_key
        FROM [EdxDW].[edx].[FactCourseStart] CS
        JOIN [EdxDW].[edx].[DimUser] U
        ON CS.UserKey = U.UserKey
        JOIN [EdxDW].[edx].[DimUserPII] PII
        ON PII.UserKey = U.UserKey
        JOIN [EdxDW].[edx].[DimCourse] C
        ON CS.CourseKey = C.CourseKey
        WHERE C.CourseRunId = '{}'
        AND IsBadRow = 0
        ORDER BY DateKey
        """.format(course_id)

        COURSE_COMPLETE_QUERY = """
        SELECT U.UserId as user_id
            ,CC.[DateKey] as date_key
        FROM [EdxDW].[edx].[FactCourseCompletion] CC
        JOIN [EdxDW].[edx].[DimUser] U
        ON CC.UserKey = U.UserKey
        JOIN [EdxDW].[edx].[DimUserPII] PII
        ON PII.UserKey = U.UserKey
        JOIN [EdxDW].[edx].[DimCourse] C
        ON CC.CourseKey = C.CourseKey
        WHERE C.CourseRunId = '{}'
        AND IsBadRow = 0
        ORDER BY DateKey
        """.format(course_id)

        COURSE_DATES_QUERY = """
        SELECT TOP(1) [CourseRunStartDate],[CourseRunEndDate]
        FROM [EdxDW].[edx].[DimCourse] C
        WHERE C.[CourseRunId] = '{}'
        """.format(course_id)

        LOG.info('Querying clickstream event data...')
        events = pd.read_sql(EVENTS_QUERY, CONN)
        LOG.info('Done')

        LOG.info('Querying forum data...')
        forums = pd.read_sql(FORUMS_QUERY, CONN)
        LOG.info('Done')

        LOG.info('Querying course starts data...')
        course_starts = pd.read_sql(COURSE_START_QUERY, CONN)
        LOG.info('Done')

        LOG.info('Querying course completions data...')
        course_completions = pd.read_sql(COURSE_COMPLETE_QUERY, CONN)
        LOG.info('Done')

        LOG.info('Querying course dates data...')
        course_dates = pd.read_sql(COURSE_DATES_QUERY, CONN)
        course_start_date, course_end_date = get_course_dates(course_dates)
        LOG.info('Done')

        LOG.info('Cleaning up raw sql data...')

        events.columns = [
            'user_id', 'event_type', 'event_source', 'course_id',
            'event_grade', 'event_attempts', 'event_max_grade', 'event_sub_correct',
            'event_time'
        ]
        events['user_id'] = events['user_id'].astype('int64')
        print(events.head())

        forums.columns = [
            'title', 'comment_text', 'author_id', 'votes_up', 'votes_down',
            'votes_count', 'votes_point', 'comment_count', 'parent_id', 'comment_thread_id',
            'course_id', 'text_type', 'update_timestamp'
        ]
        print(forums.head())

        LOG.info('Done')

        # Only include events for users that have "started" the course
        # according to Data Warehouse definitions
        LOG.info('Merging events with course starts...')
        events = pd.merge(course_starts, events, how='inner', on='user_id')
        LOG.info('Done')

        LOG.info('Filtering events')
        events = filter_events(events, course_start_date)
        print(events.head())
        LOG.info('Done')

        LOG.info('Saving data to csv...')
        save_df_to_csv(events, 'events', course_id)
        save_df_to_csv(forums, 'forums', course_id)
        save_df_to_csv(course_starts, 'course_starts', course_id)
        save_df_to_csv(course_completions, 'course_completions', course_id)
        save_df_to_csv(course_dates, 'course_dates', course_id)
        LOG.info('Done')

    return (events, forums, course_starts, course_completions, course_start_date, course_end_date)


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


def filter_events(events_df, course_start_date):
    """
    1. Filter events to relevant (event_type)s
    2. Define the course_week of the event based on event_time
    3. Set correct problem attempts to a new event_type to
       aggregate on.
    """

    events_to_capture = [
        'problem_check',
        'play_video'
    ]

    events_sub = events_df.loc[events_df['event_type'].isin(events_to_capture),:]
    events_sub = events_sub.drop(
        events_sub.loc[(events_sub['event_type'] == 'problem_check') & (events_sub['event_source'] == 'browser'),:].index
    )
    SUBSECTION_VIEWED_MARKER = 'subsection_viewed'

    events_df.loc[events_df['event_type'].str.contains('/courseware/'),'event_type'] = SUBSECTION_VIEWED_MARKER
    subsection_events = events_df.loc[events_df['event_type'] == SUBSECTION_VIEWED_MARKER,:]
    events_sub = events_sub.fillna(0.0)
    events_ = events_sub.append(subsection_events)
    events_ = events_.reset_index(drop=True)

    events_['course_week'] = events_['event_time'].apply(
        lambda date_string: course_week(date_string, course_start_date)
    )
    events_.loc[events_['event_sub_correct'] == 'true', 'event_type'] = 'problem_check:correct'

    return events_
