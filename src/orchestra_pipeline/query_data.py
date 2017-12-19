# -*- coding: utf-8 -*-
import luigi
from luigi.util import inherits
# from pipeline_v2 import Params
from common import ADLTarget, MSSqlConnection
from orchestra_pipeline import util


class RawUserActivity(luigi.Task):
    """
    Tasks that implement this mixin must return data in a specific format
    for events, course_starts, course_completions, forum posts and course start/end dates
    """
    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()
    course_start_date = luigi.DateParameter()
    data_origin = luigi.ChoiceParameter(choices=['edx'], var_type=str, default='edx')

    def requires(self):
        if self.data_origin == 'edx':
            return {
                'events': EventsQueryTask(course_id=self.course_id, course_week=self.course_week, course_start_date=self.course_start_date),
                'forums': ForumsQueryTask(course_id=self.course_id, course_week=self.course_week),
                'course_starts': CourseStartsQueryTask(course_id=self.course_id, course_week=self.course_week),
                'course_completions': CourseCompletionsQueryTask(course_id=self.course_id, course_week=self.course_week)
            }

    def run(self):
        pass
    
    def output(self):
        return self.input()
    

class EventsQueryTask(luigi.Task):

    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()
    course_start_date = luigi.DateParameter()

    _query = """
        SELECT
            UserId, EventType, EventSource, CourseId,
            EventGrade, EventAttempts, EventMaxGrade, 
            EventSub_Correct, EventTime
        FROM [EdxStaging].[edx].[Edx_DailyEvents]
        WHERE (Host = 'courses.edx.org' and CourseId = '{}')
        AND EventTime > '{}'
        AND UserId IS NOT NULL
    """

    def output(self):
        return ADLTarget('data/{}/week_{}/events.csv'.format(self.course_id, self.course_week))

    def run(self):
        conn = MSSqlConnection()
        events = conn.run_query(self._query.format(self.course_id, util.two_weeks_ago()))
        with self.output().open('w') as output:
            events.columns = [
                'user_id', 'event_type', 'event_source', 'course_id',
                'event_grade', 'event_attempts', 'event_max_grade', 
                'event_sub_correct', 'event_time'
            ]
            events['user_id'] = events['user_id'].astype('int64')
            events = self.filter_events(events, self.course_start_date)
            events.to_csv(output, index=False)

    def filter_events(self, events_df, course_start_date):
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
            lambda date_string: util.course_week(date_string, course_start_date)
        )
        events_.loc[events_['event_sub_correct'] == 'true', 'event_type'] = 'problem_check:correct'

        return events_


class ForumsQueryTask(luigi.Task):

    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()

    _query = """
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
    """

    def output(self):
        return ADLTarget('data/{}/week_{}/forums.csv'.format(self.course_id, self.course_week))

    def run(self):
        conn = MSSqlConnection()
        forums = conn.run_query(self._query.format(self.course_id))
        with self.output().open('w') as output:
            forums.columns = [
                'title', 'comment_text', 'author_id', 'votes_up', 'votes_down',
                'votes_count', 'votes_point', 'comment_count', 'parent_id', 'comment_thread_id',
                'course_id', 'text_type', 'update_timestamp'
            ]
            forums.to_csv(output, index=False)


class CourseStartsQueryTask(luigi.Task):

    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()

    _query = """
        SELECT 
            U.UserId as user_id
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
    """

    def output(self):
        return ADLTarget('data/{}/week_{}/course_starts.csv'.format(self.course_id, self.course_week))

    def run(self):
        conn = MSSqlConnection()
        res = conn.run_query(self._query.format(self.course_id))
        with self.output().open('w') as output:
            res.to_csv(output, index=False)


class CourseCompletionsQueryTask(luigi.Task):

    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()

    _query = """
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
    """

    def output(self):
        return ADLTarget('data/{}/week_{}/course_completions.csv'.format(self.course_id, self.course_week))

    def run(self):
        conn = MSSqlConnection()
        res = conn.run_query(self._query.format(self.course_id))
        with self.output().open('w') as output:
            res.to_csv(output, index=False)


class CourseDatesQueryTask(luigi.Task):

    course_id = luigi.Parameter()

    _query = """
        SELECT TOP(1) [CourseRunStartDate],[CourseRunEndDate]
        FROM [EdxDW].[edx].[DimCourse] C
        WHERE C.[CourseRunId] = '{}'
    """

    def output(self):
        return ADLTarget('data/{}/course_dates.csv'.format(self.course_id))

    def run(self):
        conn = MSSqlConnection()
        res = conn.run_query(self._query.format(self.course_id))
        with self.output().open('w') as output:
            res.to_csv(output, index=False)

if __name__ == "__main__":
    luigi.run()