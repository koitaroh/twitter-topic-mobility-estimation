import datetime
import os
import pymysql
import configparser
import sqlalchemy

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

# Database configuration
conf = configparser.ConfigParser()
conf.read('config.cfg')
REMOTE_DB = {
    "host": conf.get('RDS', 'host'),
    "user": conf.get('RDS', 'user'),
    "passwd": conf.get('RDS', 'passwd'),
    "db_name": conf.get('RDS', 'db_name'),
}
ENGINE_CONF = "mysql+pymysql://" + REMOTE_DB["user"] + ":" + REMOTE_DB["passwd"] + "@" + REMOTE_DB["host"] + "/" + \
              REMOTE_DB["db_name"] + "?charset=utf8mb4"
TABLE_NAME = "social_activity_201307_1"

DATA_CONFIGURATION = {
    'ENGINE_CONF': "mysql+pymysql://" + REMOTE_DB["user"] + ":" + REMOTE_DB["passwd"] + "@" + REMOTE_DB["host"] + "/" + REMOTE_DB["db_name"] + "?charset=utf8mb4",
    'TABLE_NAME': "social_activity_201307_1"
}

# Experiment parameters
EXPERIMENT_NAME = "Experiment_20170104_1549"
TIMESTART = "2013-07-25 00:00:00"
TIMEEND = "2013-07-31 23:59:59"
TIMESTART_TEXT = TIMESTART[0:10]
TIMEEND_TEXT = TIMEEND[0:10]
# Area of interest
# aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
AOI = [138.72, 34.9, 140.87, 36.28]  # Greater Tokyo Area
UNIT_TEMPORAL = 60  # in minutes
UNIT_SPATIAL_METER = 1000
NUM_TOPIC = 10
# sample size (originally: 22571472)
SAMPLE_SIZE = 1000000

EXPERIMENT_PARAMETERS = {
    'EXPERIMENT_NAME': "Experiment_20170104_1549",
    'TIMESTART' : "2013-07-25 00:00:00",
    'TIMEEND' : "2013-07-31 23:59:59",
    'TIMESTART_TEXT' : TIMESTART[0:10],
    'TIMEEND_TEXT' : TIMEEND[0:10],
    # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
    'AOI' : [138.72, 34.9, 140.87, 36.28],
    'UNIT_TEMPORAL' : 60,  # in minutes
    'UNIT_SPATIAL_METER' : 1000,
    'NUM_TOPIC' : 10,
    # sample size (originally: 22571472)
    'SAMPLE_SIZE' : 1000000
}

# Experiment resource files
## Local
FIGURE_DIR = "/Users/koitaroh/Dropbox/Figures"
EXPERIMENT_DIR = "/Users/koitaroh/Documents/Data/Experiments/"
GPS_DIR = "/Users/koitaroh/Documents/Data/GPS/2013/"
STOPLIST_FILE = "../data/stoplist_jp.txt"

# Remote
# FIGURE_DIR = "/Users/koitaroh/Dropbox/Figures"
# EXPERIMENT_DIR = "/home/ubuntu/Experiments/"
# GPS_DIR = "/home/ubuntu/data/GPS/"
# # MALLET_FILE = "/Users/koitaroh/Documents/GitHub/Mallet/bin/mallet"
# STOPLIST_FILE = "../data/stoplist_jp.txt"



if EXPERIMENT_NAME == '':
    now = datetime.datetime.now()
    EXPERIMENT_NAME = "Experiment_" + now.strftime("%Y%m%d_%H%M%S")
MODELS_DIR = EXPERIMENT_DIR + EXPERIMENT_NAME
logger.info("EXPERIMENT_NAME: %s", EXPERIMENT_NAME)
logger.info("MODELS_DIR: %s", MODELS_DIR)
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

MODEL_NAME = "ClassifiedTweet_" + TIMESTART_TEXT + '_' + TIMEEND_TEXT
TEXTS_DIR = os.path.join(MODELS_DIR, MODEL_NAME)
STOPLIST = set(line.strip() for line in open(STOPLIST_FILE, encoding='UTF-8'))
DOCNAMES = os.path.join(MODELS_DIR, MODEL_NAME + "_docnames.csv")
TWEET_DF_NAME = os.path.join(MODELS_DIR, MODEL_NAME + ".csv")
TWEET_COUNTER_FILE = os.path.join(MODELS_DIR, "tweet_counter.pkl")
GPS_COUNTER_FILE = os.path.join(MODELS_DIR, "gps_counter.pkl")
LSI_counter_file_1 = os.path.join(MODELS_DIR, MODEL_NAME + "_lsi.pkl")
LDA_counter_file_1 = os.path.join(MODELS_DIR, MODEL_NAME + "_lda.pkl")
OUTDIR = os.path.join(MODELS_DIR, MODEL_NAME + "/")
