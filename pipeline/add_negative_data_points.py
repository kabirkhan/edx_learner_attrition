import logging
import pandas as pd
import numpy as np
from pipeline.util import *
from tqdm import tqdm

LOG_CONFIG = {'root':{'handlers':('console', 'file'), 'level':'DEBUG'}}

FORMAT = '%(name)s:   %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
LOG = logging.getLogger(__name__)


def add_neg_data_points(course_id, features, from_checkpoint=False):
    """
    Add negative data points to a features dataframe
    """
    if from_checkpoint:
        def get_data_from_file(name):
            return pd.read_csv("{}/{}/{}.csv".format(get_data_path(), course_id, name))
        # data = get_data_from_file('model_data')
        data = get_data_from_file('model_data_l')
    else:
        LOG.info('Adding negative data points for course: {}'.format(course_id))

        data = features.copy()

        user_ids = pd.unique(data['user_id'])
        course_weeks = set(pd.unique(data['course_week']))

        users_completed_count = 0
        # model_data_path = '{}/{}/model_data.csv'.format(get_data_path(), course_id)

        for user_id in tqdm(user_ids):
            user_rows = data.loc[(data['user_id'] == user_id), :]

            active_course_weeks = set(pd.unique(user_rows.loc[:, 'course_week']))
            zero_course_weeks = list(course_weeks - active_course_weeks)

            started_week = user_rows.loc[:, 'user_started_week'].min()
            last_active_week = user_rows.loc[:, 'user_last_active_week'].min()
            completed_week = user_rows.loc[:, 'user_completed_week'].min()

            # zero_course_weeks = [x for x in zero_course_weeks if (x < last_active_week and x > 0)]
            last_uncompleted_active_week = last_active_week if completed_week < 0 else completed_week
            zero_course_weeks = [x for x in zero_course_weeks if (x < last_uncompleted_active_week and x > started_week)]

            zero_rows_df = pd.DataFrame(
                np.zeros((len(zero_course_weeks), len(data.columns))),
                columns=data.columns
            )

            zero_rows_df.loc[:, 'user_id'] = user_id
            zero_rows_df.loc[:, 'course_week'] = zero_course_weeks
            zero_rows_df.loc[:, 'user_started_week'] = started_week
            zero_rows_df.loc[:, 'user_last_active_week'] = last_active_week
            zero_rows_df.loc[:, 'user_completed_week'] = completed_week

            data = data.append(zero_rows_df)
            data = data.reset_index(drop=True)
            
        data = data.sort_values(['user_id', 'course_week']).reset_index(drop=True)
        data = data.fillna(0.0)

        LOG.info('Done')
        LOG.info('Calculating user dropped out next week.')
        try:
            data = calculate_drop_out_next_week(data)
        except Exception as ex:
            LOG.error(ex)
        LOG.info('Done')

        LOG.info('Saving to csv...')
        # save_df_to_file(data, 'model_data', course_id)
        save_df_to_file(data, 'model_data_l', course_id)
        LOG.info('Done')
        del data
        

def calculate_drop_out_next_week(features):
    """
    Calculate user_dropped_out_next_week value (this is the value we will try to predict)
    """
    data_ = features.copy()

    # Default to 0
    data_['user_dropped_out_next_week'] = 0.

    feature_cols = [
        'num_video_plays', 'num_problems_attempted',
        'num_problems_correct', 'num_forum_posts'
    ]

    # If the user has no engagment activity for a week, mark them as "dropped out next week" (1)
    ranked = data_[feature_cols]
    data_.loc[ranked[ranked == 0].dropna().index, 'user_dropped_out_next_week'] = 1
    # Calculate if user was active at all in the previous week
    shifted = data_.set_index(['user_id', 'course_week']).groupby(level=0).shift(1)[feature_cols]
    shifted.loc[shifted[shifted != 0].dropna(thresh=1).index, 'user_active_previous_week'] = 1
    shifted['user_active_previous_week'] = shifted['user_active_previous_week'].fillna(0.)
    shifted = shifted.reset_index().drop(columns=feature_cols)
    data_ = pd.merge(data_, shifted, on=['user_id', 'course_week'])

    # If user active previous week, mark them as "not dropped out next week" (0)
    # even if they have no activity.
    # This gives the user a 1 week buffer. 
    # i.e. we consider a user dropped out only after they have been inactive for two weeks.
    data_.loc[data_['user_active_previous_week'] == 1, 'user_dropped_out_next_week'] = 0

    # The user actually dropped out here so mark them as "dropped out next week" (1)
    data_.loc[data_['user_last_active_week'] == data_['course_week'], 'user_dropped_out_next_week'] = 1

    # If the user completes the course but keeps engaging, make sure the user
    # is marked as "not dropped out next week" (0)
    data_.loc[data_['user_completed_week'] > -1, 'user_dropped_out_next_week'] = 0

    # swap column names so user_dropped_out_next_week is the last column
    col_names = list(data_.columns[:-2]) + ['user_active_previous_week', 'user_dropped_out_next_week']
    data_ = data_[col_names]

    return data_
    