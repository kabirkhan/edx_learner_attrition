from datetime import datetime
import luigi
from luigi.util import requires
import pandas as pd
import numpy as np
from tqdm import tqdm
from pipeline.util import course_week
from pipeline.build_features import BuildFeatures
from common import ADLTarget


@requires(BuildFeatures)
class AddNegativeDataPoints(luigi.Task):
    """
    Task that adds negative data points to the features data frame
    for inactive course weeks per user
    """

    course_id = luigi.Parameter()
    course_week = luigi.IntParameter()
    course_start_date = luigi.DateParameter()

    def output(self):
        return ADLTarget('data/{}/week_{}/model_data.csv'.format(
            self.course_id, self.course_week
        ))

    def run(self):
        with self.input()['features'].open() as features_file:
            features = pd.read_csv(features_file)
            data = self.add_neg_data_points(self.course_id, features)
        with self.output().open('w') as output:
            data.to_csv(output, index=False)

    def add_neg_data_points(self, course_id, features):
        """
        Add negative data points to a features dataframe
        """
        data = features.copy()
        user_ids = pd.unique(data['user_id'])
        course_weeks = set(pd.unique(data['course_week']))

        for user_id in tqdm(user_ids):
            user_rows = data.loc[(data['user_id'] == user_id), :]

            active_course_weeks = set(pd.unique(user_rows.loc[:, 'course_week']))
            zero_course_weeks = list(course_weeks - active_course_weeks)

            started_week = user_rows.loc[:, 'user_started_week'].min()
            last_active_week = user_rows.loc[:, 'user_last_active_week'].min()
            completed_week = user_rows.loc[:, 'user_completed_week'].min()

            zero_course_weeks = [x for x in zero_course_weeks if (x < last_active_week and x > 0)]

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

        # LOG.info('Done')
        # LOG.info('Calculating user dropped out next week.')
        data = self.calculate_drop_out_next_week(data)
        # LOG.info('Done')

        return data

    def calculate_drop_out_next_week(self, features):
        """
        Calculate user_dropped_out_next_week value (this is the value we will try to predict)
        """
        data_ = features.copy()
        data_['user_dropped_out_next_week'] = np.where(
            data_['user_last_active_week'] == data_['course_week'], 1, 0
        )
        data_.loc[data_['user_completed_week'] > -1, 'user_dropped_out_next_week'] = 0
        return data_

if __name__ == "__main__":
    luigi.run()