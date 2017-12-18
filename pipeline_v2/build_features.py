from datetime import datetime, timedelta
import luigi
from luigi.util import requires
import pandas as pd
# from pipeline_v2 import Params
from pipeline_v2.query_data import RawUserActivity, CourseDatesQueryTask
from pipeline_v2.util import course_week
from textblob import TextBlob
from pipeline_v2.adl_luigi import ADLTarget

@requires(RawUserActivity)
class BuildFeatures(luigi.Task):
    """
    Converts raw sql data into initial features dataframe
    """
    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()
    course_start_date = luigi.DateParameter()

    def output(self):
        return {
            'features': ADLTarget('data/{}/week_{}/features.csv'.format(self.course_id, self.course_week)),
        }

    def run(self):
        """
        Build features dataframe 
        """
        input = self.input()
        
        events = self.get_df_from_input('events')
        print(events.head())
        forums = self.get_df_from_input('forums')
        course_starts = self.get_df_from_input('course_starts')
        course_completions = self.get_df_from_input('course_completions')
        
        features = self.build_features(self.course_id, events, forums, course_starts, course_completions, self.course_start_date)
        features = features[features['course_week'] >= 1]
        features_target = self.output().get('features')
        with features_target.open(mode='w') as output:
            features.to_csv(output, index=False)

    def get_df_from_input(self, key):
        with self.input().get(key).open() as raw_input:
            return pd.read_csv(raw_input)

    def build_features(self, 
                       course_id,
                       events,
                       forums,
                       course_starts,
                       course_completions,
                       course_start_date):
        """
        Build a features dataframe by aggregating events for (user, course_week)
        """

        # LOG.info('Building features...')

        event_features = self._build_events_features(events)
        event_features.columns = event_features.columns.get_level_values(0)

        forum_features = self._build_forum_features(forums, course_start_date)
        # LOG.info('Done')

        # LOG.info('Merging events features and forum features')
        features = pd.merge(event_features, forum_features, how='left', on=['user_id', 'course_week'])
        # LOG.info('Done')
        print(features.columns)
        print(features.head())

        features.columns = [
            'course_week',
            'user_id',
            'num_video_plays',
            'num_problems_incorrect',
            'num_problems_correct',
            'num_subsections_viewed',
            'num_forum_posts',
            'num_forum_votes',
            'avg_forum_sentiment'
        ]

        # Merge features with course_starts and course_completions
        # LOG.info('Merging features with coures starts and completions')
        data = pd.merge(features, course_starts, how='inner', on='user_id')
        data.columns = list(data.columns)[0:-1] + ['user_started_date_key']
        data = pd.merge(data, course_completions, how='left', on='user_id')
        data.columns = list(data.columns)[0:-1] + ['user_completed_date_key']
        # LOG.info('Done.')

        def fill_date_column(column_name):
            """
            Convert string column with dates formatted like 20170101 into
            datetime objects
            """
            data[column_name] = data[column_name].astype(str)
            data.loc[data[column_name] == 'nan', column_name] = datetime.strftime(
                course_start_date - timedelta(days=8), '%Y%m%d'
            )
            return data

        data = fill_date_column('user_started_date_key')
        data = fill_date_column('user_completed_date_key')

        data['user_started_week'] = data['user_started_date_key'].apply(
            lambda x: course_week(x, course_start_date)
        )
        data['user_completed_week'] = data['user_completed_date_key'].apply(
            lambda x: course_week(x, course_start_date)
        )

        last_active_week = data.sort_values('course_week').groupby('user_id').last()[['course_week']]
        last_active_week.columns = ['user_last_active_week']
        last_active_week = last_active_week.reset_index()
        data = pd.merge(data, last_active_week, how='left', on='user_id')

        data['num_problems_attempted'] = data['num_problems_incorrect'] + data['num_problems_correct']
        data = data[[
            'user_id',
            'course_week',
            'num_video_plays',
            'num_problems_attempted',
            'num_problems_correct',
            'num_subsections_viewed',
            'num_forum_posts',
            'num_forum_votes',
            'avg_forum_sentiment',
            'user_started_week',
            'user_last_active_week',
            'user_completed_week'
        ]]

        # LOG.info(data.columns)

        return data

    def _build_events_features(self, events_df):
        """
        Aggregate events to create grouped feature DataFrame
        """
        features = pd.DataFrame(
            events_df.groupby(['course_week', 'user_id'])['event_type'].value_counts()
        )        
        features.columns = ['event_counts']
        features = features.unstack()
        features = features.fillna(0.0)
        features = features.reset_index()        
        return features

    def _build_forum_features(self, forums_df, course_start_date):
        forums_ = self._process_forum_data(forums_df, course_start_date)

        forum_features = forums_.groupby(['course_week', 'author_id'])[
            ['comment_text', 'votes_count', 'sentiment']
        ].agg({
            'comment_text': 'count',
            'votes_count': 'sum',
            'sentiment': 'mean'
        }).reset_index()

        forum_features.columns = ['course_week', 'user_id', 'comment_text', 'votes_count', 'sentiment']

        return forum_features

    def _process_forum_data(self, forums_, course_start_date):
        forums_df = forums_.copy()
        forums_df['course_week'] = forums_df['update_timestamp'].apply(
            lambda x: course_week(x, course_start_date)
        )
        forums_df = forums_df[['author_id', 'comment_text', 'votes_count', 'course_week']]

        forums_df['sentiment'] = forums_df['comment_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )

        return forums_df


if __name__ == "__main__":
    luigi.run()