import sys
from datetime import datetime

import click
import pipeline

# parameter handling
@click.command()
@click.option(
    '--course-id',
    help='ID for an edX Course Run. (e.g. Microsoft+DAT206x+3T2017)'
)
@click.option(
    '--from-checkpoint',
    help='If True, pull data from csv files in the /data folder. False by default',
    is_flag=True,
    default=False
)
def run(course_id, from_checkpoint):

    """
    Send notification message to an existing service bus queue
    """

    if not course_id:
        print('ERROR: course_id is a required parameter')
        sys.exit()
    print('Starting Pipeline for Course: {}'.format(course_id))
    if from_checkpoint:
        print('Using cached data files from data/*.csv')

    start = datetime.now()

    data = pipeline.query_data(course_id, from_checkpoint=from_checkpoint)
    features = pipeline.build_features(course_id, *data, from_checkpoint=from_checkpoint)
    model_data = pipeline.add_neg_data_points(course_id, features, from_checkpoint=from_checkpoint)
    # data = pipeline.query_data(course_id, from_checkpoint=True)
    # features = pipeline.build_features(course_id, *data, from_checkpoint=True)
    # model_data = pipeline.add_neg_data_points(course_id, features, from_checkpoint=True)
    preds, accuracy, confusion_matrix = pipeline.fit_score_predict(course_id, from_checkpoint=from_checkpoint)
    delta = datetime.now() - start
    end = round(delta.seconds / 60)
    print('Finished Pipeline for Course: {}. Run finished in {} minutes'.format(course_id, end))

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
