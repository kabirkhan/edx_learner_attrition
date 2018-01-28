from collections import Counter
import numpy as np
import pandas as pd

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import keras
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.cross_validation import train_test_split
from pipeline.util import *


# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', hidden_layers_dim=8, num_hidden_layers=2):
    # create model
    model = Sequential()
    model.add(Dense(hidden_layers_dim, input_dim=9))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    for i in range(num_hidden_layers):
        model.add(Dense(hidden_layers_dim))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        
    model.add(Dense(hidden_layers_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_data(current_course_id):
    """
    TODO Fix how this training data is sampled
    e.g. bootstrap sampling of a random number of courses
    to get a total of > 1 million training samples
    """

    # train = pd.read_csv('{}/{}/model_data.csv'.format(get_data_path(), 'Microsoft+DAT206x+3T2017'))
    train = None
    past_course_ids = [f for f in os.listdir(get_data_path()) if not f.startswith('.')]
    try:
        past_course_ids.remove(current_course_id)
    except ValueError:
        print('Not in list')

    for course_id in past_course_ids:
        if '4T2017' not in course_id:
            try:
                # course_run_data = pd.read_csv('{}/{}/model_data.csv'.format(get_data_path(), course_id))
                path = '{}/{}/model_data_l.csv'.format(get_data_path(), course_id)
                course_run_data = pd.read_csv(path)
            except Exception:
                print('model_data.csv does not exist for course: ', course_id)
                continue
                # pass
            if train is None:
                train = course_run_data
            else:
                train = train.append(course_run_data)

    print('Training data done.')

    train = train.reset_index(drop=True)
    # test = pd.read_csv('{}/{}/model_data.csv'.format(get_data_path(), current_course_id))
    test = pd.read_csv('{}/{}/model_data_l.csv'.format(get_data_path(), current_course_id))


    X_cols = [
        'course_week', 'num_video_plays', 'num_problems_attempted',
        'num_problems_correct', 'num_subsections_viewed', 'num_forum_posts',
        'num_forum_votes', 'avg_forum_sentiment', 'user_started_week',
    ]

    scaler = StandardScaler()
    scaler.fit(train[X_cols])

    X_train = scaler.transform(train[X_cols])
    X_test = scaler.transform(test[X_cols])

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    y_train = np.array(train['user_dropped_out_next_week']).astype(np.float32)
    y_test = np.array(test['user_dropped_out_next_week']).astype(np.float32)

    return (X_train, y_train, X_test, y_test)


def fit_score_predict(course_id, train=False):

    print('GETTING DATA: ')
    X_train, y_train, X_test, y_test = get_data(course_id)
    print('Done.')

    batch_size = 50
    
    if not train:
        model = load_model('model.h5')
    else:
        model = create_model()

        print('Fitting model')
        y_counts = Counter(y_train)
        positive_upweight = (y_counts[0] / y_counts[1])

        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        cv_scores = []
        for train, val in kfold.split(X_train, y_train):
            model.fit(X_train,
                    y_train, 
                    epochs=10, 
                    batch_size=batch_size, 
                    class_weight={ 0: 1., 1: 2 }, 
                    validation_split=0.1,
                    verbose=2)
        print('Done')
        try:
            model.save('model.h5')
        except:
            print('FAILED TO SAVE MODEL')

    print('Evaluating model on data for course: {}'.format(course_id))

    score = model.evaluate(X_test, y_test, batch_size)
    print('Model score', score)

    preds = model.predict(X_test, batch_size)
    final_preds = np.round(preds)

    print('PREDS: ', final_preds[0:10])
    print('Y_TEST: ', y_test[0:10])

    conf_matrix = metrics.confusion_matrix(y_test, final_preds)

    tn, fp, fn, tp = conf_matrix.ravel()
    total = len(y_test)
    final_acc = (tn + tp) / total

    test_data_orig = pd.read_csv('{}/{}/model_data_l.csv'.format(get_data_path(), course_id))
    test_data_orig['predicted_user_dropped_out_next_week'] = final_preds

    pred_pivot = _create_pivot_table(test_data_orig, 'predicted_user_dropped_out_next_week')
    real_pivot = _create_pivot_table(test_data_orig, 'user_dropped_out_next_week')

    # save_df_to_file(pred_pivot, 'predicted_dropouts', course_id, type='excel')
    # save_df_to_file(real_pivot, 'real_dropouts', course_id, type='excel')
    # save_df_to_file(test_data_orig, 'model_data_with_preds', course_id)

    save_df_to_file(pred_pivot, 'predicted_dropouts_l', course_id, type='excel')
    save_df_to_file(real_pivot, 'real_dropouts_l', course_id, type='excel')
    save_df_to_file(test_data_orig, 'model_data_with_preds_l', course_id)

    print('ACCURACY: ', final_acc)
    print('CONFUSION MATRIX: ')
    print(conf_matrix)
    print(conf_matrix / len(y_test))

    return (final_preds, final_acc, conf_matrix)

def _create_pivot_table(df, val_col):
    df_pivot = df.pivot_table(
        index='user_id', columns=['course_week'], values=val_col, fill_value=-1
    )
    df_colored = df_pivot.style.applymap(_cell_colors)
    return df_colored

def _cell_colors(s):
    ret = 'background-color: {}'
    if s == 0:
        ret = ret.format('#228b22')
    elif s == 1:
        ret = ret.format('#dc143c')
    else:
        ret = ret.format('#d3d3d3')

    return ret
