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
        data = get_data_from_file('model_data')
    else:
        LOG.info('Adding negative data points for course: {}'.format(course_id))

        data = features.copy()
        user_ids = pd.unique(data['user_id'])
        course_weeks = sorted(pd.unique(data['course_week']))

        for user_id in tqdm(user_ids):
            for week in course_weeks:
                exists = data.loc[(data['user_id'] == user_id) & (data['course_week'] == week), :]
                user_info = data.loc[(data['user_id'] == user_id), :]
                started_week = list(user_info['user_started_week'])[0]
                last_active_week = list(user_info['user_last_active_week'])[0]
                if len(exists) == 0 and (week <= last_active_week):
                    new_zero_row = pd.DataFrame([{
                        'user_id': user_id,
                        'course_week': week,
                        'user_started_week': started_week,
                        'user_last_active_week': last_active_week,
                        'user_completed_week': list(user_info['user_completed_week'])[0],
                    }], columns=data.columns).fillna(0.0)
                    data = data.append(new_zero_row)
                    data = data.reset_index(drop=True)

        data = data.sort_values(['user_id', 'course_week']).reset_index(drop=True)

        LOG.info('Done')
        LOG.info('Calculating user dropped out next week.')
        data = calculate_drop_out_next_week(features)
        LOG.info('Done')

        LOG.info('Saving to csv...')
        save_df_to_csv(data, 'model_data', course_id)
        LOG.info('Done')

    return data


def calculate_drop_out_next_week(features):
    """
    Calculate user_dropped_out_next_week value (this is the value we will try to predict)
    """
    data_ = features.copy()
    data_['user_dropped_out_next_week'] = np.where(
        data_['user_last_active_week'] == data_['course_week'], 1, 0
    )
    data_.loc[data_['user_completed_week'] > -1, 'user_dropped_out_next_week'] = 0
    return data_
    