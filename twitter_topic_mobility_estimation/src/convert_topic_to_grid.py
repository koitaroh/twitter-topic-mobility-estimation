__author__ = 'koitaroh'

# convert_topic_to_grid.py
# Last Update: 2016-08-13
# Author: Satoshi Miyazawa
# koitaroh@gmail.com
# Convert tweets to population density array p[t, x, y]

import csv
import numpy
import pickle
import os
import datetime
import sqlalchemy
import gensim

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


def define_spatiotemporal_unit(timestart, timeend, aoi, unit_temporal, unit_spatial):
    x_start = int(aoi[0]/unit_spatial)
    # logger.debug(x_start)
    y_start = int(aoi[1]/unit_spatial)
    # logger.debug(y_start)
    x_end = int(aoi[2]/unit_spatial)
    # logger.debug(x_end)
    y_end = int(aoi[3]/unit_spatial)
    # logger.debug(y_end)
    x_size = (x_end - x_start) + 1
    # logger.debug(x_size)
    y_size = (y_end - y_start) + 1
    # logger.debug(y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = (t_end - t_start) / datetime.timedelta(minutes=unit_temporal)
    # logger.debug(t_size)
    t_size = int(t_size) + 1
    # logger.debug(t_size)
    return [t_size, x_size, y_size, 10]


def raw_txy_to_index_txy(timestart, timeend, aoi, unit_temporal, unit_spatial, t, x, y):
    timestart = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    t_index = int((t - timestart)/datetime.timedelta(minutes=unit_temporal))
    x_index = int((x - aoi[0])/unit_spatial)
    y_index = int((y - aoi[1])/unit_spatial)
    return [t_index, x_index, y_index]

def refer_doctopic_matrix(model, vector):
    problist = []
    problist = model[vector]
    # logger.debug(problist)
    return problist


def topic_to_grid(st_units, model, outfile):
    density = numpy.zeros(st_units)
    with open(DOCNAMES, encoding="utf-8") as f:
        f.readline()  # read one line in order to skip the header
        i = 0
        for line in f:
            datehour, longitude, latitude = line.rstrip().split('_')
            datestring = datehour[0:10]
            houstring = datehour[11:13]
            t = datestring + " " + houstring + ":00:00"
            x = int(longitude)*0.01
            y = int(latitude)*0.01
            vector = corpus[i]
            # print(vector)
            try:
                if len(vector):
                    index = raw_txy_to_index_txy(timestart, timeend, aoi, unit_temporal, unit_spatial, t, x, y)
                    problist = refer_doctopic_matrix(model, vector)
                    # print(problist)
                    # read topic
                    for j in problist:
                        if j[0] < 10:
                            # logger.debug(j)
                            # logger.debug(index)
                            density[index[0], index[1], index[2], j[0]] = j[1]
            except IndexError as err:
                logger.debug(t, err)
                continue
            finally:
                i += 1
    output = open(outfile, 'wb')
    logger.debug("saving.")
    pickle.dump(density, output, protocol=4)
    output.close()


if __name__ == '__main__':
    logger.debug('initializing.')

    data_dir = "/Users/koitaroh/Documents/Data/GPS/2013/"
    # tweet_counter_file = '/Users/koitaroh/Documents/Data/Tweet/counter_20130701_20130731.pkl'

    # Retrieval parameters for testing
    timestart = "2013-07-31 12:00:00"
    timestart_text = "20130731"
    timeend = "2013-07-31 23:59:59"
    timeend_text = "20130731"
    # Area of interest
    # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
    aoi = [139.71, 35.65, 139.77, 35.69]  # Greater Tokyo Area

    # # Retrieval parameters
    # timestart = "2013-07-01 00:00:00"
    # timestart_text = "20130701"
    # timeend = "2013-07-31 23:59:59"
    # timeend_text = "20130731"
    # # Area of interest
    # # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
    # aoi = [138.72, 34.9, 140.87, 36.28]  # Greater Tokyo Area

    unit_temporal = 60  # in minutes
    unit_spatial = 0.01  # in degrees
    example_x = 138.750000
    example_y = 34.91000
    example_t = "2013-07-03 00:00:00"

    # Parameter setting
    MALLET_FILE = '/Users/koitaroh/Documents/GitHub/Workspace/twitter_topic_mobility_prediction/mallet-2.0.7/bin/mallet'
    TEXTS_DIR = "/Users/koitaroh/Documents/Data/Tweet/ClassifiedTweet_20130731_20130731"
    MODELS_DIR = "/Users/koitaroh/Documents/Data/Model"
    MODEL_NAME = "ClassifiedTweet_20130731_20130731"
    STOPLIST_FILE = "/Users/koitaroh/Documents/Data/Model/stoplist_jp.txt"
    stoplist = set(line.strip() for line in open(STOPLIST_FILE))
    DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME+"_docnames.csv")
    LSI_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_lsi_20130731_20130731.pkl'
    LSI_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_lsi_20130731_20130731_2.pkl'
    LDA_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_lda_20130731_20130731.pkl'
    LDA_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_lda_20130731_20130731_2.pkl'
    HDP_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_hdp_20130731_20130731.pkl'
    HDP_counter_file = '/Users/koitaroh/Documents/Data/Model/counter_hdp_20130731_20130731_2.pkl'
    MODEL_LSI = os.path.join(MODELS_DIR, MODEL_NAME+".lsi")
    MODEL_LDA = os.path.join(MODELS_DIR, MODEL_NAME+".lda")
    MODEL_HDP = os.path.join(MODELS_DIR, MODEL_NAME+".hdp")
    corpus = gensim.corpora.MmCorpus(os.path.join(MODELS_DIR, MODEL_NAME+"_mt.mm"))
    # lsi_model = numpy.load(MODEL_LSI, allow_pickle=True)
    lsi_model = gensim.models.LsiModel.load(MODEL_LSI)
    lda_model = gensim.models.LdaModel.load(MODEL_LDA)
    hdp_model = gensim.models.HdpModel.load(MODEL_HDP)

    st_units = define_spatiotemporal_unit(timestart, timeend, aoi, unit_temporal, unit_spatial)
    logger.debug(st_units)
    # # example_index = raw_txy_to_index_txy_degree(timestart, timeend, aoi, unit_temporal, unit_spatial, example_t, example_x, example_y)
    # # logger.debug(example_index)

    topic_to_grid(st_units, lsi_model, LSI_counter_file)
    topic_to_grid(st_units, lda_model, LDA_counter_file)
    topic_to_grid(st_units, hdp_model, HDP_counter_file)

    # tweets_to_grid(st_units, tweet_counter_file)
    # logger.debug("Done.")
