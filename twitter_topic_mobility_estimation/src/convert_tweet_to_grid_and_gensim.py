# convert_tweet_to_gensim.py
# Last Update: 2016-05-26
# Author: Satoshi Miyazawa
# koitaroh@gmail.com


import collections
import datetime
import logging
import os
import pickle
from logging import handlers

import numpy
import pandas
import sqlalchemy

import convert_points_to_grid
import settings as s
import utility_database

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


def create_twitter_text_files_metric(table, st_units, tweet_counter_file, aoi, timestart, timeend, tweet_df_name, outdir, unit_temporal, unit_spatial, x_unit_degree, y_unit_degree):
    logger.debug('Loading tweets to dataframe.')
    density = numpy.zeros(st_units)
    # Loading tweets to dataframe
    engine, conn, metadata = utility_database.establish_db_connection_mysql_twitter_ssh()
    # engine = sqlalchemy.create_engine(engine_conf, echo=False)
    sql = "SELECT * FROM %s where (tweeted_at between '%s' and '%s') and (x BETWEEN %s and %s) and (y BETWEEN %s and %s) and (words != '')" %(table, timestart, timeend, aoi[0], aoi[2], aoi[1], aoi[3])
    # conn = engine.connect()
    tweet_df = pandas.read_sql_query(sql, engine)
    # logger.debug('Applying indices.')
    # Applying indices
    # tweet_df_time = tweet_df['tweeted_at']
    # temp = pandas.DatetimeIndex(tweet_df['tweeted_at'])
    # tweet_df['date'] = temp.date
    # tweet_df['time'] = temp.time
    # tweet_df['t_index'] = (tweet_df['date']).astype(str) + '-' + (tweet_df['time'].astype(str)).str[:2]
    # tweet_df['x_index'] = (tweet_df['x']/0.01).astype(int)
    # tweet_df['y_index'] = (tweet_df['y']/0.01).astype(int)
    # logger.debug('Saving files.')
    # # Save files
    # tweet_df.to_csv(tweet_df_name, encoding='utf-8')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # tweet_counter = collections.Counter()
    for index, row in tweet_df.iterrows():

        # t_index = str(row['t_index'])
        # x_index = str(row['x_index'])
        # y_index = str(row['y_index'])
        # words = str(row['words'])
        # workfile = t_index + "_" + x_index + "_" + y_index + ".txt"
        # tweet_counter[t_index + "_" + x_index + "_" + y_index] += 1
        # f = open(outdir + workfile, 'a')
        # f.write(words + "\n")
        # f.close()

        t = str(row['tweeted_at'])
        x = float(row['x'])
        y = float(row['y'])
        words = str(row['words'])
        try:
            # index = convert_points_to_grid.raw_txy_to_index_txy_metric(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
            index = convert_points_to_grid.raw_txy_to_index_txy_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y, x_unit_degree, y_unit_degree)
            workfile = str(index[0]) + "_" + str(index[1]) + "_" + str(index[2]) + ".txt"
            f = open(outdir + workfile, 'a')
            f.write(words + "\n")
            f.close()

            density[index[0], index[1], index[2]] += 1

        except IndexError as err:
            logger.debug(t, err)
            continue
    f = open(tweet_counter_file, 'wb')
    logger.debug("saving numpy array.")
    pickle.dump(density, f)
    f.close()


def create_twitter_text_files_degree(engine_conf, table, st_units, tweet_counter_file, aoi, timestart, timeend, tweet_df_name, outdir, unit_temporal, unit_spatial):
    logger.debug('Loading tweets to dataframe.')
    density = numpy.zeros(st_units)
    # Loading tweets to dataframe
    engine = sqlalchemy.create_engine(engine_conf, echo=False)
    sql = "SELECT * FROM %s where (tweeted_at between '%s' and '%s') and (x BETWEEN %s and %s) and (y BETWEEN %s and %s) and (words != '')" %(table, timestart, timeend, aoi[0], aoi[2], aoi[1], aoi[3])
    conn = engine.connect()
    tweet_df = pandas.read_sql_query(sql, engine)
    logger.debug('Applying indices.')
    # Applying indices
    tweet_df_time = tweet_df['tweeted_at']
    temp = pandas.DatetimeIndex(tweet_df['tweeted_at'])
    tweet_df['date'] = temp.date
    tweet_df['time'] = temp.time
    tweet_df['t_index'] = (tweet_df['date']).astype(str) + '-' + (tweet_df['time'].astype(str)).str[:2]
    tweet_df['x_index'] = (tweet_df['x']/0.01).astype(int)
    tweet_df['y_index'] = (tweet_df['y']/0.01).astype(int)
    logger.debug('Saving files.')
    # Save files
    tweet_df.to_csv(tweet_df_name, encoding='utf-8')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    tweet_counter = collections.Counter()
    for index, row in tweet_df.iterrows():
        t_index = str(row['t_index'])
        x_index = str(row['x_index'])
        y_index = str(row['y_index'])
        words = str(row['words'])
        workfile = t_index + "_" + x_index + "_" + y_index + ".txt"
        tweet_counter[t_index + "_" + x_index + "_" + y_index] += 1
        f = open(outdir + workfile, 'a')
        f.write(words + "\n")
        f.close()

        t = str(row['tweeted_at'])
        x = float(row['x'])
        y = float(row['y'])
        try:
            index = convert_points_to_grid.raw_txy_to_index_txy_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
            density[index[0], index[1], index[2]] += 1
        except IndexError as err:
            logger.debug(t, err)
            continue
    f = open(tweet_counter_file, 'wb')
    logger.debug("saving numpy array.")
    pickle.dump(density, f)
    f.close()


if __name__ == '__main__':
    now = datetime.datetime.now()
    logger.debug('Setting parameters.')


    engine_conf = s.ENGINE_CONF
    table = s.TABLE_NAME

    timestart = s.TIMESTART
    timestart_text = s.TIMESTART_TEXT
    timeend = s.TIMEEND
    timeend_text = s.TIMEEND_TEXT
    aoi = s.AOI
    unit_temporal = s.UNIT_TEMPORAL
    # Change denpending on metric or degree
    unit_spatial = s.UNIT_SPATIAL_METER
    num_topic = s.NUM_TOPIC

    tweet_counter_file = s.TWEET_COUNTER_FILE

    MODELS_DIR = s.MODELS_DIR
    MODEL_NAME = s.MODEL_NAME
    # MALLET_FILE = s.MALLET_FILE
    TEXTS_DIR = s.TEXTS_DIR
    STOPLIST = s.STOPLIST
    DOCNAMES = s.DOCNAMES
    LSI_counter_file = s.LSI_counter_file_1
    LDA_counter_file = s.LDA_counter_file_1
    HDP_counter_file = s.HDP_counter_file_1
    tweet_df_name = s.TWEET_DF_NAME
    outdir = s.OUTDIR

    # Change function in gps_to_grid_metric depending on metric or degree
    # st_units = convert_points_to_grid.define_spatiotemporal_unit_degree(aoi, timestart, timeend, unit_spatial, unit_temporal)
    # create_twitter_text_files_degree(engine_conf, table, st_units, tweet_counter_file, aoi, timestart, timeend, tweet_df_name, outdir, unit_temporal, unit_spatial)

    # Metric
    # st_units, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_metric(aoi, timestart, timeend, unit_spatial, unit_temporal)
    st_units, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial,
                                                                        unit_temporal)
    create_twitter_text_files_metric(engine_conf, table, st_units, tweet_counter_file, aoi, timestart, timeend,
                                     tweet_df_name, outdir, unit_temporal, unit_spatial, x_unit_degree, y_unit_degree)

