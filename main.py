import pipeline
import click
import sys

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

    data = pipeline.query_data(course_id, from_checkpoint)
    model_data = pipeline.build_features(course_id, *data, from_checkpoint=from_checkpoint)
    print(model_data.head())

if __name__ == "__main__":
    run() # pylint: disable=no-value-for-parameter
