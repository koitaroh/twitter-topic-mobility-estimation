import os

import matplotlib.cm as cm
import numpy

import convert_gps_to_grid
import convert_points_to_grid
import convert_tweet_to_grid_and_gensim
import run_gensim_topicmodels
import run_regression
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
    DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME+"_1_docnames.csv")
    EXPERIMENT_NAME = s.EXPERIMENT_NAME
    NUM_TOPIC = s.NUM_TOPIC

    TABLE_NAME = s.TABLE_NAME
    tweet_df_name = s.TWEET_DF_NAME
    outdir = s.OUTDIR

    LSI_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lsi.pkl")
    LDA_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_lda.pkl")
    # HDP_counter_file = os.path.join(MODELS_DIR, MODEL_NAME + "_hdp.pkl")
    # ESTIMATION_LSI_SVR_FILE = os.path.join(MODELS_DIR, "estimation_svr_lsi.pkl")
    ESTIMATION_LSI_GB_FILE = os.path.join(MODELS_DIR, "estimation_gb_lsi.pkl")


    # Convert_gps_to_grid
    # Change function in gps_to_grid_metric depending on metric or degree
    st_units, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal)
    convert_gps_to_grid.gps_to_grid_metric_people_new(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, x_unit_degree, y_unit_degree, gps_dir, gps_counter_file)


    # Convert_tweet_to_grid_and_gensim
    # Change function in gps_to_grid_metric depending on metric or degree
    st_units_topic, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_topic_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, NUM_TOPIC)
    convert_tweet_to_grid_and_gensim.create_twitter_text_files_metric(TABLE_NAME, st_units, TWEET_COUNTER_FILE, aoi, timestart, timeend,
                                                                      tweet_df_name, outdir, unit_temporal, unit_spatial, x_unit_degree, y_unit_degree)

    # run_gensim_topicmodels
    run_gensim_topicmodels.run_gensim_topicmodels_metric(NUM_TOPIC, st_units_topic, TEXTS_DIR, stoplist, MODELS_DIR, MODEL_NAME, LSI_counter_file, LDA_counter_file)

    # run_regression
    tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter = \
        run_regression.load_data(TWEET_COUNTER_FILE, gps_counter_file, LSI_counter_file, LDA_counter_file, NUM_TOPIC)
    run_regression.run_regression(tweet_counter_row, gps_counter_row, tweet_lsi_counter, tweet_lda_counter)
    run_regression.create_estimation(TWEET_COUNTER_FILE, LSI_counter_file, ESTIMATION_LSI_GB_FILE, tweet_lsi_counter, gps_counter_row)
    logger.info('Done.')