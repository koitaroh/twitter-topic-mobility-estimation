import os
import csv
import datetime
import pickle
import json
import math
import gzip
import numpy
import pandas
import gensim
import MeCab
import sqlalchemy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import plotly.plotly as py  # tools to communicate with Plotly's server
from sklearn import datasets, linear_model, metrics, ensemble
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score

import settings as s

numpy.set_printoptions(edgeitems=10)

# Logging ver. 2016-07-12
from logging import handlers
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('log.log', maxBytes=1000000, backupCount=3)  # file handler
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()  # console handler
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Initializing %s', __name__)

colors = [cm.RdBu(0.85), cm.RdBu(0.7), cm.PiYG(0.7), cm.Spectral(0.38), cm.Spectral(0.25)]


# def k_fold_cross_validation(X, y, eval_size):
#     kf = KFold(len(y), round(1. / eval_size), shuffle=True)
#     train_indices, valid_indices = next(iter(kf))
#     X_train, y_train = X[train_indices], y[train_indices]
#     X_valid, y_valid = X[valid_indices], y[valid_indices]
#     return X_train, X_valid, y_train, y_valid


def logistic_regression(tweet_X_train, tweet_X_test, gps_y_train, gps_y_test):
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    param_grid = {'C': [1, 10, 100] }
    logger.info("Logistic Regression")
    # logit = grid_search.GridSearchCV(linear_model.LogisticRegression(solver='newton-cg', max_iter=200, multi_class='multinomial', penalty='l2'),
    #                                  param_grid=param_grid)

    logit = linear_model.LogisticRegression(solver='newton-cg', max_iter=100, multi_class='multinomial', penalty='l2', C=1)
    gps_y_train = numpy.ravel(gps_y_train)
    # gps_y_train = gps_y_train.astype(int)
    logit.fit(tweet_X_train, gps_y_train)
    predLogit = logit.predict(tweet_X_test)
    print('Coefficients: \n', logit.coef_)
    # The mean square error
    print("RMSE: %.2f"
          % math.sqrt(numpy.mean((logit.predict(tweet_X_test) - gps_y_test) ** 2)))
    print('Variance score: %.2f' % logit.score(tweet_X_test, gps_y_test))
    return predLogit


def svr_cv(tweet_x, gps_y):
    logger.info("SVR")
    # param_grid = [{'kernel':['rbf', 'sigmoid'], 'C':[0.001, 0.01, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.00001], 'tol': [0.0001, 0.001]}]
    param_grid = [{'kernel':['rbf'], 'C':[1000], 'tol': [0.0001, 0.001]}]
    svr = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1)
    svr.fit(tweet_x, gps_y)
    logger.info(svr.best_estimator_)
    scores = cross_val_score(svr, tweet_x, gps_y, cv=3)
    logger.info("R^2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    svr = GridSearchCV(SVR(), param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
    svr.fit(tweet_x, gps_y)
    logger.info(svr.best_estimator_)
    scores = cross_val_score(svr, tweet_x, gps_y, cv=3)
    logger.info("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return True




def gb_cv(tweet_x, gps_y):
    logger.info("GB with cross validation")
    # param_grid = [{'loss': ['ls','lad','huber'],'n_estimators': [300, 500], 'max_depth': [3, 4], 'min_samples_split': [1],'learning_rate': [0.01], 'max_features': [None]}]
    param_grid = [{'loss': ['ls'],'n_estimators': [300, 500], 'max_depth': [3, 4], 'min_samples_split': [2],'learning_rate': [0.01], 'max_features': [None]}]
    gb = GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid=param_grid, n_jobs=-1)
    gb.fit(tweet_x,gps_y)
    logger.info(gb.best_estimator_)
    scores = cross_val_score(gb, tweet_x, gps_y, cv=3)
    logger.info("R^2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    gb = GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1)
    gb.fit(tweet_x, gps_y)
    logger.info(gb.best_estimator_)
    scores = cross_val_score(gb, tweet_x, gps_y, cv=3)
    logger.info("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return True



def get_nonzero_arrays(x_array, y_array):
    nonzero_idx_x = numpy.nonzero(x_array)
    y_array = y_array[nonzero_idx_x[0]]
    x_array = x_array[nonzero_idx_x[0], :]
    nonzero_idx_y = numpy.nonzero(y_array)
    y_array = y_array[nonzero_idx_y[0]]
    x_array = x_array[nonzero_idx_y[0], :]

    logger.info('Shape of x:  %s', x_array.shape)
    return x_array, y_array

# def get_nonzero_arrays_all(y_array, x_array, array_2, array_3):
#     nonzero_idx_x = numpy.nonzero(x_array)
#     y_array = y_array[nonzero_idx_x[0]]
#     x_array = x_array[nonzero_idx_x[0], :]
#     array_2 = array_2[nonzero_idx_x[0], :]
#     array_3 = array_3[nonzero_idx_x[0], :]
#
#
#     logger.info('Shape of x:  %s', x_array.shape)
#     return x_array, y_array, array_2, array_3

def get_nonzero_arrays_all(x_array, y_array, array_2, array_3):
    logger.info('Shape of original x:  %s', x_array.shape)
    nonzero_idx_x = numpy.nonzero(x_array)
    y_array_1 = y_array[nonzero_idx_x[0]]
    x_array_1 = x_array[nonzero_idx_x[0], :]
    array_2_1 = array_2[nonzero_idx_x[0], :]
    array_3_1 = array_3[nonzero_idx_x[0], :]
    logger.info('Shape of nonzero-x x:  %s', x_array_1.shape)
    nonzero_idx_y = numpy.nonzero(y_array_1)
    y_array_2 = y_array_1[nonzero_idx_y[0]]
    x_array_2 = x_array_1[nonzero_idx_y[0], :]
    array_2_2 = array_2_1[nonzero_idx_y[0], :]
    array_3_2 = array_3_1[nonzero_idx_y[0], :]
    logger.info('Shape of nonzero-y x:  %s', x_array_2.shape)
    # nonzero_idx_2 = numpy.nonzero(array_2_2)
    # y_array_3 = y_array_2[nonzero_idx_2[0]]
    # x_array_3 = x_array_2[nonzero_idx_2[0], :]
    # array_2_3 = array_2_2[nonzero_idx_2[0], :]
    # array_3_3 = array_3_2[nonzero_idx_2[0], :]
    # logger.info('Shape of x:  %s', x_array_3.shape)
    # nonzero_idx_3 = numpy.nonzero(array_3_3)
    # y_array_4 = y_array_3[nonzero_idx_3[0]]
    # x_array_4 = x_array_3[nonzero_idx_3[0], :]
    # array_2_4 = array_2_3[nonzero_idx_3[0], :]
    # array_3_4 = array_3_3[nonzero_idx_3[0], :]
    # logger.info('Shape of x:  %s', x_array_4.shape)
    return x_array_2, y_array_2, array_2_2, array_3_2


def get_sample_array_all(array_0, array_1, array_2, array_3):
    if len(array_0) > SAMPLE_SIZE:
        sample_idx = numpy.random.choice(numpy.arange(len(array_0)), SAMPLE_SIZE, replace=False)
    else:
        sample_idx = numpy.random.choice(numpy.arange(len(array_0)), len(array_0), replace=False)
    array_0 = array_0[sample_idx, :]
    array_1 = array_1[sample_idx]
    array_2 = array_2[sample_idx, :]
    array_3 = array_3[sample_idx, :]
    logger.info('Shape of sample x:  %s', array_1.shape)
    return array_0, array_1, array_2, array_3


def create_whole_out_sample(x_array, y_array):
    sample_idx = numpy.random.choice(numpy.arange(len(x_array)), 20000, replace=False)

    x_array = x_array[sample_idx, :]
    y_array = y_array[sample_idx]
    y_array = numpy.ravel(y_array)

    # Create train and test set
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(x_array)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(y_array)

    nCases = len(x_array)
    nTrain = int(numpy.floor(nCases * 0.8))
    # Split the data into training/testing sets
    tweet_X_train = x_array[:nTrain]  # Test with 20% data
    # tweet_X_train = tweet_X_train.astype(int)
    tweet_X_test = x_array[nTrain:]
    # tweet_X_test = tweet_X_test.astype(int)
    # print(tweet_X_train.shape)
    # print(tweet_X_test.shape)

    # Split the targets into training/testing sets
    gps_y_train = y_array[:nTrain]
    gps_y_test = y_array[nTrain:]
    # print(gps_y_train.shape)
    # print(gps_y_test.shape)

    return tweet_X_train, tweet_X_test, gps_y_train, gps_y_test

def run_regression_models(x_array, y_array):
    y_array = numpy.ravel(y_array)
    # x_array, y_array = get_nonzero_arrays(x_array, y_array)
    # y_array = numpy.ravel(y_array)
    # tweet_X_train, tweet_X_test, gps_y_train, gps_y_test = create_whole_out_sample(x_array, y_array)
    # svr(tweet_X_train, tweet_X_test, gps_y_train, gps_y_test)
    # gb_regression(tweet_X_train, tweet_X_test, gps_y_train, gps_y_test)

    # Cross validations
    svr_cv(x_array, y_array)
    gb_cv(x_array, y_array)


    # # Plot joint result
    # plt.hold('on')
    # plt.plot(gps_y_test, gps_y_test, label='true data')
    # plt.plot(gps_y_test, predSvr, 'ro', label='SVR')
    # plt.plot(gps_y_test, predGb, 'yo', label='GB')
    # plt.xlabel('Test')
    # plt.ylabel('Prediction')
    # plt.title('Regression performance')
    # plt.legend()
    # plt.show()
    return None


def create_estimation(x_original_file_1, x_original_file_2, y_prediction_file, x_sample_row, y_sample_row):
    logger.info('Loading data for estimation.')
    x_original_1 = numpy.load(x_original_file_1, allow_pickle=True)
    x_original_2 = numpy.load(x_original_file_2, allow_pickle=True)
    stshape = x_original_1.shape
    stlength = stshape[0] * stshape[1] * stshape[2]
    x_original_1_row = x_original_1.reshape(stlength, 1)
    x_original_2_row = x_original_2.reshape(stlength, NUM_TOPIC)
    x_original_12_row = numpy.hstack((x_original_1_row, x_original_2_row))

    y_sample_row = numpy.ravel(y_sample_row)

    logger.info("Fitting GB with cross validation")
    # param_grid = [{'loss': ['ls','lad','huber'],'n_estimators': [300, 500], 'max_depth': [3, 4], 'min_samples_split': [1],'learning_rate': [0.01], 'max_features': [None]}]
    param_grid = [{'loss': ['ls'],'n_estimators': [300, 500], 'max_depth': [3, 4], 'min_samples_split': [2],'learning_rate': [0.01], 'max_features': [None]}]
    gb = GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid=param_grid, n_jobs=-1)
    # gb = grid_search.GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid=param_grid, scoring='mean_squared_error', n_jobs=-1)
    gb.fit(x_sample_row, y_sample_row)
    logger.info(gb.best_estimator_)
    scores = cross_val_score(gb, x_sample_row, y_sample_row, cv=3)
    logger.info("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    logger.info("Creating extimation.")
    y_prediction_row = gb.predict(x_original_12_row)
    y_prediction = y_prediction_row.reshape(stshape)

    output = open(y_prediction_file, 'wb')
    logger.debug("saving.")
    pickle.dump(y_prediction, output, protocol=4)
    return None


def load_data(TWEET_COUNTER_FILE, GPS_COUNTER_FILE, LSI_counter_file, LDA_counter_file, NUM_TOPIC):
    tweet_counter = numpy.load(TWEET_COUNTER_FILE, allow_pickle=True)
    logger.info('Shape of tweet_counter:  %s', tweet_counter.shape)
    gps_counter = numpy.load(GPS_COUNTER_FILE, allow_pickle=True)
    logger.info('Shape of gps_counter:  %s', gps_counter.shape)
    lsi_counter = numpy.load(LSI_counter_file, allow_pickle=True)
    lda_counter = numpy.load(LDA_counter_file, allow_pickle = True)
    logger.info("Reshape to new row vectors")
    stshape = tweet_counter.shape
    stlength = stshape[0] * stshape[1] * stshape[2]
    tweet_counter_row = tweet_counter.reshape(stlength, 1)
    gps_counter_row = gps_counter.reshape(stlength, 1)
    lsi_counter_row = lsi_counter.reshape(stlength, NUM_TOPIC)
    lda_counter_row = lda_counter.reshape(stlength, NUM_TOPIC)
    # hdp_counter_row = hdp_counter.reshape(22074480, 10)
    tweet_lsi_counter = numpy.hstack((tweet_counter_row, lsi_counter_row))
    tweet_lda_counter = numpy.hstack((tweet_counter_row, lda_counter_row))
    # tweet_lda_counter_multiplied = lda_counter_row * tweet_counter_row
    # tweet_hdp_counter = numpy.hstack((tweet_counter_row, hdp_counter_row))
    # tweet_hdp_counter_multiplied = hdp_counter_row * tweet_counter_row
    tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter = get_nonzero_arrays_all(tweet_counter_row,
                                                                                                      gps_counter_row,
                                                                                                      tweet_lsi_counter,
                                                                                                      tweet_lda_counter)
    tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter = get_sample_array_all(tweet_counter_row,
                                                                                                    gps_counter_row,
                                                                                                    tweet_lsi_counter,
                                                                                                    tweet_lda_counter)
    return tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter


def run_regression(tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter):
    logger.info("Running tweet -> gps")
    run_regression_models(tweet_counter_row, gps_counter_row)
    logger.info("Running tweet + LSA -> gps")
    run_regression_models(tweet_lsi_counter, gps_counter_row)
    logger.info("Running tweet + LDA -> gps")
    run_regression_models(tweet_lda_counter, gps_counter_row)
    return None


SAMPLE_SIZE = s.SAMPLE_SIZE
NUM_TOPIC = s.NUM_TOPIC

if __name__ == '__main__':

    # Parameter setting
    FIGURE_DIR = s.FIGURE_DIR
    # MALLET_FILE = s.MALLET_FILE
    TEXTS_DIR = s.TEXTS_DIR
    MODELS_DIR = s.MODELS_DIR
    GPS_DIR = s.GPS_DIR
    MODEL_NAME = s.MODEL_NAME
    STOPLIST_FILE = s.STOPLIST_FILE
    stoplist = set(line.strip() for line in open(STOPLIST_FILE))
    DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME+"_1_docnames.csv")
    EXPERIMENT_NAME = s.EXPERIMENT_NAME
    # stshape = [1488, 215, 138, 10]
    # stlength = stshape[0] * stshape[1] * stshape[2]
    # STLENGTH = 22074480

    GPS_COUNTER_FILE = os.path.join(MODELS_DIR, "gps_counter.pkl")
    TWEET_COUNTER_FILE = os.path.join(MODELS_DIR, "tweet_counter.pkl")

    LSI_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lsi.pkl")
    LDA_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lda.pkl")
    HDP_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_hdp.pkl")

    # PREDICTION_LSI_SVR_FILE = os.path.join(MODELS_DIR, "/prediction_svr_lsi.pkl")
    ESTIMATION_LSI_GB_FILE = os.path.join(MODELS_DIR, "estimation_gb_lsi.pkl")

    # Main function
    tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter = \
        load_data(TWEET_COUNTER_FILE, GPS_COUNTER_FILE, LSI_counter_file, LDA_counter_file, NUM_TOPIC)
    run_regression(tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter)
    # create_estimation(TWEET_COUNTER_FILE, LSI_counter_file, ESTIMATION_LSI_GB_FILE, tweet_lsi_counter, gps_counter_row)
