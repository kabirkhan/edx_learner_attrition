from collections import Counter
import numpy as np
import pandas as pd
import argparse
import os

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dense, Dropout, Input, Average
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import plot_model
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_data_path():
    return './data'


def get_data(current_course_id):
    """
    Fetch data as train/test split and normalize
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
        'num_forum_votes', 'avg_forum_sentiment', 
        'user_started_week', 'user_active_previous_week'
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


# Function to create model, required for KerasClassifier
def create_model(model_input, hidden_layers_conf=[], name=''):
    # create model
    model = None
    
    for i, layer in enumerate(hidden_layers_conf):
        
        if i == 0:            
            model = Dense(layer['n_units'])(model_input)            
                
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(layer.get('dropout', 0.2))(model)
    
    model = Dense(1)(model)
    model = BatchNormalization()(model)
    predictions = Activation('sigmoid')(model)
    
    model = Model(inputs=model_input, outputs=predictions)
    if name:
        model.name = name

    return model


def compile_and_train(model, x_train, y_train, optimizer='adam', metrics=['accuracy'], num_epochs=20, batch_size=100): 
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics) 
    filepath = 'weights/' + model.name + '.{epoch:02d}-{loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', period=1
    )
    
    history = model.fit(x=x_train,
                        y=y_train, 
                        batch_size=batch_size, 
                        epochs=num_epochs, 
                        verbose=1,
                        class_weight={ 0: 1., 1: 2 },
                        callbacks=[checkpoint])

    return history


def ensemble_models(models, model_input):
    # collect outputs of models in a list
    model_outputs = [model.outputs[0] for model in models]

    # averaging outputs
    avg = Average()(model_outputs)

    # build model from same input and avg output
    model_ens = Model(inputs=model_input, outputs=avg, name='ensemble') 
   
    return model_ens


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


def false_neg_metric(y_true, y_pred):
    """
    Calculate ratio of false negatives in predictions with sklearn conf_matrix metric
    """
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    false_neg_ratio = conf_matrix[1][0]
    return false_neg_ratio / 100


def cohens_kappa_metric(y_true, y_pred):
    """
    Calculate Cohens Kappa with sklearn metric
    """
    return metrics.cohen_kappa_score(y_true, y_pred)


def run_model(course_id, train, num_epochs, batch_size, positive_upweight, learning_rate, layers_config_filename, outputdir):
    """
    Run model with parsed configuration
    """

    print('GETTING DATA: ')
    X_train, y_train, X_test, y_test = get_data(course_id)
    print('Done.')

    input_shape = (X_train.shape[1],)
    model_input = Input(shape=input_shape)
    
    models = []
    best_model = None
    best_model_score = -np.inf
    batch_size = 256
    
    if not train:
        current_date_string = datetime.strftime(datetime.today(), '%Y-%m-%d')
        model = load_model('model-{}.h5'.format('2018-01-23'))
    else:
        adam = optimizers.Adam(lr=learning_rate)

        with open(layers_config_filename, 'r') as f:
            import json
            layers_conf = json.loads(f.read())["layers"]

        print('Fitting model')
        
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        
        for i, (train_ind, val_ind) in enumerate(kfold.split(X_train, y_train)):

            model = create_model(model_input, 
                                 hidden_layers_conf=layers_conf, 
                                 name='kfold-{}'.format(i))

            model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])         
            
            history = model.fit(x=X_train,
                                y=y_train, 
                                batch_size=batch_size, 
                                epochs=num_epochs, 
                                verbose=2,
                                class_weight={ 0: 1., 1: positive_upweight },
                                validation_data=(X_train[val_ind], y_train[val_ind]))

            y_val_pred = model.predict(X_train[val_ind])
            y_val_true = y_train[val_ind]

            final_recall = metrics.recall_score(y_val_true, y_val_pred)
            final_acc = metrics.recall_score(y_val_true, y_val_pred)
            final_score = final_recall - final_acc

            if final_score > best_model_score:
                best_model_score = final_score
                best_model = model

            models.append(model)

    try:        
        best_model.save(os.path.join(outputdir, 'model.h5'))
    except:
        print('FAILED TO SAVE MODEL')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    print('STARTING')

    parser.add_argument('--course-id', help='Course Id to run on', required=True, default=None)
    parser.add_argument('--train', help='Train model or not', action='store_true', required=False, default=True)
    parser.add_argument('--num-epochs', help='Number of epochs to run for', required=False, type=int, default=10)
    parser.add_argument('--batch-size', help='Batch Size for train/test', required=False, type=int, default=256)
    parser.add_argument('--positive-upweight', help='How much to upweight positive preds during optimization', type=float, required=False, default=2)
    parser.add_argument('--lr', help='Learning rate for Adam Optimizer', required=False, type=float, default=0.01)
    parser.add_argument('--layers-config-file', help='JSON config file for layers', required=True, default=None)
    parser.add_argument('--outputdir', help='Directory to save best models to', required=True, default=None)

    args = vars(parser.parse_args())
    print('ARGS ALL GOOD', args)
    
    run_model(args['course_id'], 
             args['train'], 
             args['num_epochs'], 
             args['batch_size'], 
             args['positive_upweight'], 
             args['lr'], 
             args['layers_config_file'],
             args['outputdir'])
