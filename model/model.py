import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
from pipeline.util import *

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam', num_hidden_layers=2, hidden_layers_dim=8):
    # create model
    model = Sequential()
    model.add(Dense(hidden_layers_dim, input_dim=9, activation='relu'))
    for i in range(num_hidden_layers):
        model.add(Dense(hidden_layers_dim, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def get_data(past_course_ids, current_course_id):
    train = None
    for course_id in past_course_ids:
        course_run_data = pd.read_csv('{}/{}/model_data.csv'.format(get_data_path(), course_id))
        if not train:
            train = course_run_data
        else:
            train.append(course_run_data)

    train = train.reset_index(drop=True)
    test = pd.read_csv('{}/{}/model_data.csv'.format(get_data_path(), current_course_id))

    X_cols = [
        'course_week', 'num_video_plays', 'num_problems_attempted',
        'num_problems_correct', 'num_subsections_viewed', 'num_forum_posts',
        'num_forum_votes', 'avg_forum_sentiment', 'user_started_week',
    ]

    X_train = np.array(train[X_cols])
    X_test = np.array(test[X_cols])

    y_train = np.array(train['user_dropped_out_next_week'])
    y_test = np.array(test['user_dropped_out_next_week'])

    return (X_train, y_train, X_test, y_test)


def fit_score_predict():

    model = KerasClassifier(build_fn=create_model, verbose=1)

    past_course_ids = []
    for i in range(3):
        past_course_ids.append('Microsoft+DAT206x+{}T2017'.format(i + 1))

    X_train, y_train, X_test, y_test = get_data(past_course_ids, 'Microsoft+DAT206x+4T2017')

    model.fit(X_train, y_train, epochs=10, batch_size=20)
    score = model.score(X_test, y_test)
    preds = model.predict(X_test)

    conf_matrix = metrics.confusion_matrix(y_test, preds)

    return (preds, score, conf_matrix)
