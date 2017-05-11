__author__ = 'koitaroh'

# convert_estimation_to_geojsonj.py
# Last Update: 2016-12-30
# Author: Satoshi Miyazawa
# koitaroh@gmail.com

import csv
import datetime
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



def save_topic_feature_to_geojson_timeslice(topic_array_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, outfile, slice_timestart):
    topic_array = numpy.load(topic_array_file, allow_pickle=True)
    timestart = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    slice_timestart = datetime.datetime.strptime(slice_timestart, '%Y-%m-%d %H:%M:%S')
    array_shape = topic_array.shape
    my_feature_list = []
    logger.info('Creating GeoJSON file')

    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                for k in range(array_shape[2]):
                    t = timestart + (datetime.timedelta(minutes=(unit_temporal * i)))
                    if t == slice_timestart:
                        t = datetime.datetime.strftime(t, '%Y-%m-%d %H:%M:%S')
                        x = aoi[0] + (x_unit_degree * j)
                        y = aoi[1] + (y_unit_degree * k)
                        topic_dict = {"timestamp": t}
                        highest_topic = -1
                        highest_topic_value = 0.0
                        for l in range(array_shape[3]):
                            # logger.debug(topic_array[i, j, k, l])
                            topic_dict["topic_" + str(l)] = float(topic_array[i, j, k, l])
                            if float(topic_array[i, j, k, l]) > 0:
                                if l == 0:
                                    highest_topic_value = float(topic_array[i, j, k, l])
                                else:
                                    if float(topic_array[i, j, k, l]) > highest_topic_value:
                                        highest_topic = l
                                        highest_topic_value = float(topic_array[i, j, k, l])
                        topic_dict["highest_topic"] = highest_topic
                        my_polygon = Polygon([[(x, y), (x+x_unit_degree, y), (x+x_unit_degree, y+y_unit_degree), (x, y+y_unit_degree), (x, y)]])
                        my_feature = Feature(geometry=my_polygon, properties=topic_dict)
                        my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


def save_spatial_grid_to_geojson_global_mean(estimation_array_file, ground_truth_array_file, tweet_array_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, outfile):
    # Global mean
    estimation_array = numpy.load(estimation_array_file, allow_pickle=True)
    ground_truth_array = numpy.load(ground_truth_array_file, allow_pickle=True)
    tweet_array = numpy.load(tweet_array_file, allow_pickle=True)

    zero_index = numpy.where(tweet_array == 0)
    estimation_array[zero_index] = 'nan'
    error = numpy.subtract(estimation_array, ground_truth_array)
    error = numpy.square(error)
    error = numpy.sqrt(error)
    ground_truth_array_spatial = numpy.mean(ground_truth_array, axis=0)
    tweet_array_spatial = numpy.mean(tweet_array, axis=0)
    estimation_array_spatial = numpy.nanmean(estimation_array, axis=0)
    error_spatial = numpy.nanmean(error, axis=0)
    array_shape = estimation_array_spatial.shape
    my_feature_list = []
    logger.info('Creating GeoJSON file')
    with open(outfile, 'w', encoding="utf-8") as geojson_file:
        for j in range(array_shape[0]):
            for k in range(array_shape[1]):
                x = aoi[0] + (x_unit_degree * j)
                y = aoi[1] + (y_unit_degree * k)
                estimation = estimation_array_spatial[j, k]
                ground_truth = ground_truth_array_spatial[j, k]
                num_tweet = tweet_array_spatial[j,k]
                error = error_spatial[j, k]
                if error >= 0:
                    my_polygon = Polygon([[(x, y), (x + x_unit_degree, y), (x + x_unit_degree, y + y_unit_degree),
                                           (x, y + y_unit_degree), (x, y)]])
                    my_feature = Feature(geometry=my_polygon, properties={"estimation": estimation,
                                                                          "ground_truth": ground_truth,
                                                                          "num_tweet": num_tweet,
                                                                          "error": error})
                    my_feature_list.append(my_feature)
        my_feature_collection = FeatureCollection(my_feature_list)
        dump = geojson.dumps(my_feature_collection, sort_keys=True)
        geojson_file.write(dump)
    return None


if __name__ == '__main__':
    # parameter setting
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
    # GPS_DIR = "/Volumes/KOITAROHUSB/2013/"
    # GPS_DIR = "/Volumes/koitarohHDD1/GPS/2013/MOD_ATF_ITSMONAVI/"
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

    TOPIC_GEOJSON_FILE = os.path.join(MODELS_DIR, "topic_20130725_9000.geojson")

    TOPIC_HANABI_20130725_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130725_1900.geojson")
    TOPIC_HANABI_20130726_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130726_1900.geojson")
    TOPIC_HANABI_20130727_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130727_1900.geojson")
    TOPIC_HANABI_20130728_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130728_1900.geojson")
    TOPIC_HANABI_20130729_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130729_1900.geojson")
    TOPIC_HANABI_20130730_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130730_1900.geojson")
    TOPIC_HANABI_20130731_1900 = os.path.join(MODELS_DIR, "topic_hanabi_20130731_1900.geojson")

    TOPIC_HANABI_20130725_1900_TIME = "2013-07-25 19:00:00"
    TOPIC_HANABI_20130726_1900_TIME = "2013-07-26 19:00:00"
    TOPIC_HANABI_20130727_1900_TIME = "2013-07-27 19:00:00"
    TOPIC_HANABI_20130728_1900_TIME = "2013-07-28 19:00:00"
    TOPIC_HANABI_20130729_1900_TIME = "2013-07-29 19:00:00"
    TOPIC_HANABI_20130730_1900_TIME = "2013-07-30 19:00:00"
    TOPIC_HANABI_20130731_1900_TIME = "2013-07-31 19:00:00"

    SLICE_TIMESTART = "2013-07-25 09:00:00"
    SLICE_TIMEEND = "2013-07-25 23:59:59"

    # save_spatial_grid_to_geojson(ESTIMATION_LSI_GB_FILE, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, ESTIMATION_GEOJSON_FILE)
    # save_spatial_grid_to_geojson(ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, ESTIMATION_GEOJSON_FILE)
    # save_spatial_grid_to_geojson_timeslice(ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, ESTIMATION_GEOJSON_FILE_SLICE, SLICE_TIMESTART)
    # save_spatial_grid_to_geojson_global_mean(ESTIMATION_LSI_GB_FILE, GPS_COUNTER_FILE, TWEET_COUNTER_FILE, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, ESTIMATION_GEOJSON_FILE_GLOBAL_MEAN)
    # save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_GEOJSON_FILE, SLICE_TIMESTART)

    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130725_1900, TOPIC_HANABI_20130725_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130726_1900, TOPIC_HANABI_20130726_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130727_1900, TOPIC_HANABI_20130727_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130728_1900, TOPIC_HANABI_20130728_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130729_1900, TOPIC_HANABI_20130729_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130730_1900, TOPIC_HANABI_20130730_1900_TIME)
    save_topic_feature_to_geojson_timeslice(LSI_counter_file, aoi, x_unit_degree, y_unit_degree, timestart, unit_temporal, TOPIC_HANABI_20130731_1900, TOPIC_HANABI_20130731_1900_TIME)