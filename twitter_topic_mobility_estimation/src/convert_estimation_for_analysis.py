__author__ = 'koitaroh'

# convert_estimation_for_analysis.py
# Last Update: 2016-12-30
# Author: Satoshi Miyazawa
# koitaroh@gmail.com

import csv
import datetime
import statistics
import math
import logging
import os
import pickle
import json
import geopy
import numpy
import pandas
import geojson
from geojson import Polygon, Feature, Point, FeatureCollection

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

import settings as s
import convert_points_to_grid


gps_dir = s.GPS_DIR
gps_counter_file = s.GPS_COUNTER_FILE
timestart = s.TIMESTART
timestart_text = s.TIMESTART_TEXT
timeend = s.TIMEEND
timeend_text = s.TIMEEND_TEXT
# Area of interest
# aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
aoi = s.AOI
unit_temporal = s.UNIT_TEMPORAL
# unit_spatial = s.UNIT_SPATIAL
unit_spatial = s.UNIT_SPATIAL_METER
MODELS_DIR = s.MODELS_DIR


def compare_estimation_with_num_tweet(experiment_parameters, outfile, estimation_array_file, ground_truth_array_file, tweet_array_file):
    counter = {}
    estimation_array = numpy.load(estimation_array_file, allow_pickle=True)
    ground_truth_array = numpy.load(ground_truth_array_file, allow_pickle=True)
    ground_truth_array = ground_truth_array.astype(float)
    tweet_array = numpy.load(tweet_array_file, allow_pickle=True)
    zero_index = numpy.where(tweet_array == 0)
    estimation_array[zero_index] = 'nan'
    ground_truth_array[zero_index] = 'nan'
    zero_index_2 = numpy.where(ground_truth_array == 0)
    estimation_array[zero_index_2] = 'nan'
    ground_truth_array[zero_index_2] = 'nan'

    rmse_array = numpy.subtract(ground_truth_array, estimation_array)
    rmse_array = numpy.square(rmse_array)
    rmse_array = numpy.sqrt(rmse_array)

    mpe_array = numpy.multiply(numpy.abs(numpy.true_divide(numpy.subtract(ground_truth_array, estimation_array), ground_truth_array)), 100)

    array_shape = estimation_array.shape

    logger.info('Creating counter file')
    with open(outfile, 'w', encoding="utf-8") as csvfile:
        counter_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        counter_writer.writerow(['# of tweet', 'frequency', 'estimation', 'ground_truth', 'RMSE', 'MPE'])
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                for k in range(array_shape[2]):

                    estimation = estimation_array[i, j, k]
                    ground_truth = ground_truth_array[i, j, k]
                    num_tweet = tweet_array[i, j, k]
                    rmse = rmse_array[i, j, k]
                    mpe = mpe_array[i, j, k]

                    if not math.isnan(estimation):
                        # logger.debug(mpe)
                        if num_tweet in counter:
                            counter[num_tweet][0].append(estimation)
                            counter[num_tweet][1].append(ground_truth)
                            counter[num_tweet][2].append(rmse)
                            counter[num_tweet][3].append(mpe)
                        else:
                            counter[num_tweet] = [[estimation], [ground_truth], [rmse], [mpe]]
        logger.info("Calcuating mean value")
        for l, m in counter.items():
            counter_writer.writerow([l, len(m[0]), statistics.mean(m[0]), statistics.mean(m[1]), statistics.mean(m[2]), statistics.mean(m[3])])
    return None


def compare_estimation_with_time(experiment_parameters, outfile, estimation_array_file, ground_truth_array_file, tweet_array_file):
    counter = {}
    estimation_array = numpy.load(estimation_array_file, allow_pickle=True)
    ground_truth_array = numpy.load(ground_truth_array_file, allow_pickle=True)
    tweet_array = numpy.load(tweet_array_file, allow_pickle=True)

    zero_index = numpy.where(tweet_array == 0)
    estimation_array[zero_index] = 'nan'
    rmse_array = numpy.subtract(ground_truth_array, estimation_array)
    rmse_array = numpy.square(rmse_array)
    rmse_array = numpy.sqrt(rmse_array)
    ground_truth_array_temporal = numpy.mean(ground_truth_array, axis=1)
    tweet_array_temporal = numpy.mean(tweet_array, axis=1)
    estimation_array_temporal = numpy.nanmean(estimation_array, axis=1)
    rmse_array_temporal = numpy.nanmean(rmse_array, axis=1)
    ground_truth_array_temporal = numpy.mean(ground_truth_array_temporal, axis=1)
    tweet_array_temporal = numpy.mean(tweet_array_temporal, axis=1)
    estimation_array_temporal = numpy.nanmean(estimation_array_temporal, axis=1)
    rmse_array_temporal = numpy.nanmean(rmse_array_temporal, axis=1)
    array_shape = estimation_array_temporal.shape
    logger.info(array_shape)
    logger.info('Creating counter file')
    with open(outfile, 'w', encoding="utf-8") as csvfile:
        counter_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        counter_writer.writerow(['time_id', 'timestamp', 'estimation', 'ground_truth', 'num_tweet', 'RMSE'])
        for i in range(array_shape[0]):
            estimation = estimation_array_temporal[i]
            ground_truth = ground_truth_array_temporal[i]
            num_tweet = tweet_array_temporal[i]
            rmse = rmse_array_temporal[i]
            counter_writer.writerow([i, i, estimation, ground_truth, num_tweet, rmse])
    return None


if __name__ == '__main__':
    # parameter setting
    EXPERIMENT_PARAMETERS = s.EXPERIMENT_PARAMETERS
    gps_dir = s.GPS_DIR
    gps_counter_file = s.GPS_COUNTER_FILE
    TWEET_COUNTER_FILE = s.TWEET_COUNTER_FILE
    timestart = s.TIMESTART
    timestart_text = s.TIMESTART_TEXT
    timeend = s.TIMEEND
    timeend_text = s.TIMEEND_TEXT
    aoi = s.AOI
    unit_temporal = s.UNIT_TEMPORAL
    # unit_spatial = s.UNIT_SPATIAL
    unit_spatial = s.UNIT_SPATIAL_METER
    FIGURE_DIR = s.FIGURE_DIR
    # MALLET_FILE = s.MALLET_FILE
    TEXTS_DIR = s.TEXTS_DIR
    MODELS_DIR = s.MODELS_DIR
    MODEL_NAME = s.MODEL_NAME
    STOPLIST_FILE = s.STOPLIST_FILE
    stoplist = set(line.strip() for line in open(STOPLIST_FILE))
    DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME + "_1_docnames.csv")
    EXPERIMENT_NAME = s.EXPERIMENT_NAME
    NUM_TOPIC = s.NUM_TOPIC

    ENGINE_CONF = s.ENGINE_CONF
    TABLE_NAME = s.TABLE_NAME
    tweet_df_name = s.TWEET_DF_NAME
    outdir = s.OUTDIR

    LSI_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lsi.pkl")
    LDA_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lda.pkl")
    HDP_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_hdp.pkl")
    ESTIMATION_LSI_GB_FILE = os.path.join(MODELS_DIR, "estimation_gb_lsi.pkl")

    # Experiment resource files
    ## Local
    EXPERIMENT_DIR = "/Users/koitaroh/Documents/Data/Experiments/"
    GPS_DIR = "/Users/koitaroh/Documents/Data/GPS/2013/"
    MALLET_FILE = "/Users/koitaroh/Documents/GitHub/Mallet/bin/mallet"
    # STOPLIST_FILE = "../data/stoplist_jp.txt"

    DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME + "_docnames.csv")
    TWEET_DF_NAME = os.path.join(MODELS_DIR, MODEL_NAME + ".csv")
    TWEET_COUNTER_FILE = os.path.join(MODELS_DIR, "tweet_counter.pkl")
    GPS_COUNTER_FILE = os.path.join(MODELS_DIR, "gps_counter.pkl")
    LSI_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lsi.pkl")
    LDA_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lda.pkl")
    ESTIMATION_LSI_GB_FILE = os.path.join(MODELS_DIR, "estimation_gb_lsi.pkl")

    GPS_DF_SPATIAL = os.path.join(MODELS_DIR, "df_gps_spatial.csv")
    TWEET_DF_SPATIAL = os.path.join(MODELS_DIR, "df_tweet_spatial.csv")
    ESTIMATION_DF_SPATIAL = os.path.join(MODELS_DIR, "df_estimation_spatial.csv")
    ERROR_DF_SPATIAL = os.path.join(MODELS_DIR, "df_error_spatial.csv")

    ERROR_DF_NIGHT = os.path.join(MODELS_DIR, "df_error_spatial_night.csv")
    ERROR_DF_MORNING = os.path.join(MODELS_DIR, "df_error_spatial_morning.csv")
    ERROR_DF_AFTERNOON = os.path.join(MODELS_DIR, "df_error_spatial_afternoon.csv")
    ERROR_DF_EVENING = os.path.join(MODELS_DIR, "df_error_spatial_evening.csv")

    ERROR_DF_HANABI_26 = os.path.join(MODELS_DIR, "df_error_spatial_hanabi_26.csv")
    ERROR_DF_HANABI_27 = os.path.join(MODELS_DIR, "df_error_spatial_hanabi_27.csv")
    ERROR_DF_HANABI_28 = os.path.join(MODELS_DIR, "df_error_spatial_hanabi_28.csv")

    st_units, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal)
    EXPERIMENT_PARAMETERS['X_UNIT_DEGREE'] = x_unit_degree
    EXPERIMENT_PARAMETERS['Y_UNIT_DEGREE'] = y_unit_degree

    ESTIMATION_HANABI_20130725_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130725_1900.geojson")
    ESTIMATION_HANABI_20130726_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130726_1900.geojson")
    ESTIMATION_HANABI_20130727_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130727_1900.geojson")
    ESTIMATION_HANABI_20130728_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130728_1900.geojson")
    ESTIMATION_HANABI_20130729_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130729_1900.geojson")
    ESTIMATION_HANABI_20130730_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130730_1900.geojson")
    ESTIMATION_HANABI_20130731_1900 = os.path.join(MODELS_DIR, "estimation_hanabi_20130731_1900.geojson")

    ESTIMATION_HANABI_20130725_1900_TIME = "2013-07-25 19:00:00"
    ESTIMATION_HANABI_20130726_1900_TIME = "2013-07-26 19:00:00"
    ESTIMATION_HANABI_20130727_1900_TIME = "2013-07-27 19:00:00"
    ESTIMATION_HANABI_20130728_1900_TIME = "2013-07-28 19:00:00"
    ESTIMATION_HANABI_20130729_1900_TIME = "2013-07-29 19:00:00"
    ESTIMATION_HANABI_20130730_1900_TIME = "2013-07-30 19:00:00"
    ESTIMATION_HANABI_20130731_1900_TIME = "2013-07-31 19:00:00"

    ESTIMATION_GEOJSON_FILE = os.path.join(MODELS_DIR, "estimation.geojson")
    ESTIMATION_GEOJSON_FILE_SLICE = os.path.join(MODELS_DIR, "estimation_20130725_0900.geojson")
    ESTIMATION_GEOJSON_FILE_GLOBAL_MEAN = os.path.join(MODELS_DIR, "estimation_global_mean.geojson")

    ESTIMATION_COMPARE_NUM_TWEET = os.path.join(MODELS_DIR, "estimation_to_numtweet.csv")
    ESTIMATION_COMPARE_TIME = os.path.join(MODELS_DIR, "estimation_to_time.csv")


    SLICE_TIMESTART = "2013-07-25 09:00:00"
    SLICE_TIMEEND = "2013-07-25 23:59:59"


    # save_spatial_grid_to_geojson_global_mean(ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, TWEET_COUNTER_FILE, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, ESTIMATION_GEOJSON_FILE_GLOBAL_MEAN)

    # compare_estimation_with_num_tweet(EXPERIMENT_PARAMETERS, ESTIMATION_COMPARE_NUM_TWEET, ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, TWEET_COUNTER_FILE)
    compare_estimation_with_time(EXPERIMENT_PARAMETERS, ESTIMATION_COMPARE_TIME, ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, TWEET_COUNTER_FILE)