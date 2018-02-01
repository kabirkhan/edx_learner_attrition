import sys
from datetime import datetime

import click
import pipeline

# parameter handling
# @click.command()
# @click.option(
#     '--course-id',
#     help='ID for an edX Course Run. (e.g. Microsoft+DAT206x+3T2017)'
# )
# @click.option(
#     '--from-checkpoint',
#     help='If True, pull data from csv files in the /data folder. False by default',
#     is_flag=True,
#     default=False
# )
# @click.option(
#     '--train',
#     help='If True, retrain the model. False by default',
#     is_flag=True,
#     default=False
# )
def run(course_id, from_checkpoint, run_model, train):

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

    data = pipeline.query_data(course_id, from_checkpoint)
    features = pipeline.build_features(course_id, *data, from_checkpoint)
    del data
    if from_checkpoint and not run_model:
        pipeline.add_neg_data_points(course_id, features, False)
    del features
    # data = pipeline.query_data(course_id, from_checkpoint=True)
    # features = pipeline.build_features(course_id, *data, from_checkpoint=True)
    
    if run_model:
        preds, accuracy, confusion_matrix = pipeline.fit_score_predict(
            course_id, train, 6, 256, 2, 0.01, 
            '/home/kabirkhan/Documents/ML_Experiments/1_edx_learner_attrition/edx-learner-attrition/batch_ai/params.json', 
            '/home/kabirkhan/Documents/ML_Experiments/1_edx_learner_attrition/edx-learner-attrition/saved_models')
            
        print('FINAL PREDS FOR COURSE: {}: '.format(course_id))
        print(preds, accuracy, confusion_matrix)
    delta = datetime.now() - start
    end = round(delta.seconds / 60)
    print('Finished Pipeline for Course: {}. Run finished in {} minutes'.format(course_id, end))

if __name__ == "__main__":
    # run()

    # PREPROCESS
    # all_start = datetime.now()
    
    # print('STARTING FOR PAST MPP COURSES...')
    # with open('{}/mpp_current_course_ids.csv'.format(pipeline.util.get_data_path())) as top_course_ids:
    #     for top_course_id in top_course_ids:
    #         print('Running for course: ', top_course_id.strip())

    #         # should_run = False
    #         # try:
    #         #     import pandas as pd
    #         #     pd.read_csv("./data/{}/features.csv".format(top_course_id.strip()))
    #         #     print('Already done')

    #         #     # run(top_course_id.strip(), from_checkpoint=True, run_model=False, train=False)
    #         # except Exception:
    #         #     should_run = True

    #         # if should_run:
    #         try:
    #             print('No data for course: {}. Querying DW'.format(top_course_id))
    #             run(top_course_id.strip(), from_checkpoint=False, run_model=False, train=False)
    #             run(top_course_id.strip(), from_checkpoint=True, run_model=False, train=False)
    #         except Exception as exc:
    #             print('Non-failing error: ', exc)
    #             pass
    
    # all_delta = datetime.now() - all_start
    # all_end = round(all_delta.seconds / 60)
    # print('Finished Pipeline for past MPP. Run finished in {} minutes'.format(all_end))

    # TRAIN
    all_start = datetime.now()
    print('STARTING FOR TOP 30 COURSES...')
    run('Microsoft+DAT206x+1T2018', True, True, True)
    all_delta = datetime.now() - all_start
    all_end = round(all_delta.seconds / 60)
    print('Finished Pipeline for TOP 30 COURSES. Run finished in {} minutes'.format(all_end))

    # # PREDICT
    # all_start = datetime.now()
    
    # print('STARTING FOR TOP 30 COURSES...')
    # with open('{}/top_course_ids.txt'.format(pipeline.util.get_data_path())) as top_course_ids:
    #     for top_course_id in top_course_ids:
    #         if '4T2017' in top_course_id:
    #             print('Running for course: ', top_course_id)
    #             try:
    #                 run(top_course_id.strip(), from_checkpoint=True, run_model=True, train=False) # pylint: disable=no-value-for-parameter
    #             except Exception:
    #                 pass
    # all_delta = datetime.now() - all_start
    # all_end = round(all_delta.seconds / 60)
    # print('Finished Pipeline for TOP 30 COURSES. Run finished in {} minutes'.format(all_end))
