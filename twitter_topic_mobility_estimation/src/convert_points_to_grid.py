__author__ = 'koitaroh'

# convert_points_to_grid.py
# Last Update: 2016-08-13
# Author: Satoshi Miyazawa
# koitaroh@gmail.com
# Convert points (tweets or gps) to population density array p[t, x, y]

import csv
import pickle
import os
import collections
import datetime
import sqlalchemy
import numpy
import pandas
import geopy
from geopy.distance import vincenty
import settings as s

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


def define_spatiotemporal_unit_metric(aoi, timestart, timeend, unit_spatial, unit_temporal):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], aoi[2])
    x1y2 = (aoi[3], aoi[0])
    x2y2 = (aoi[3], aoi[2])
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    logger.debug("X distance: %s meters", x_distance)
    logger.debug("Y distance: %s meters", y_distance)
    x_size = int(x_distance // unit_spatial) + 1
    y_size = int(y_distance // unit_spatial) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.info("T size: %s", t_size)
    logger.info("Spatiotemporal units: %s", [t_size, x_size, y_size])
    return [t_size, x_size, y_size]

def define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], aoi[2])
    x1y2 = (aoi[3], aoi[0])
    x2y2 = (aoi[3], aoi[2])
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    # logger.debug("X distance: %s meters", x_distance)
    # logger.debug("Y distance: %s meters", y_distance)
    logger.debug("X distance: %s meters, Y distance: %s meters", x_distance, y_distance)
    x_unit_degree = round((((aoi[2] - aoi[0]) * unit_spatial) / x_distance), 4)
    y_unit_degree = round((((aoi[3] - aoi[1]) * unit_spatial) / y_distance), 4)
    logger.debug("X unit in degree: %s degrees, Y unit in degree: %s degrees", x_unit_degree, y_unit_degree)
    x_size = int((aoi[2] - aoi[0]) // x_unit_degree) + 1
    y_size = int((aoi[3] - aoi[1]) // y_unit_degree) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.info("T size: %s", t_size)
    logger.info("Spatiotemporal units: %s", [t_size, x_size, y_size])
    return [t_size, x_size, y_size], x_unit_degree, y_unit_degree


def define_spatiotemporal_unit_topic_metric(aoi, timestart, timeend, unit_spatial, unit_temporal, num_topic):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], aoi[2])
    x1y2 = (aoi[3], aoi[0])
    x2y2 = (aoi[3], aoi[2])
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    logger.debug("X distance: %s meters", x_distance)
    logger.debug("Y distance: %s meters", y_distance)
    x_size = int(x_distance // unit_spatial) + 1
    y_size = int(y_distance // unit_spatial) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.info("T size: %s", t_size)
    logger.info("Spatiotemporal units: %s", [t_size, x_size, y_size, num_topic])
    return [t_size, x_size, y_size, num_topic]


def define_spatiotemporal_unit_topic_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, num_topic):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], aoi[2])
    x1y2 = (aoi[3], aoi[0])
    x2y2 = (aoi[3], aoi[2])
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    logger.debug("X distance: %s meters", x_distance)
    logger.debug("Y distance: %s meters", y_distance)
    logger.debug("X distance: %s meters, Y distance: %s meters", x_distance, y_distance)
    x_unit_degree = round((((aoi[2] - aoi[0]) * unit_spatial) / x_distance), 4)
    y_unit_degree = round((((aoi[3] - aoi[1]) * unit_spatial) / y_distance), 4)
    logger.debug("X unit in degree: %s degrees, Y unit in degree: %s degrees", x_unit_degree, y_unit_degree)
    x_size = int((aoi[2] - aoi[0]) // x_unit_degree) + 1
    y_size = int((aoi[3] - aoi[1]) // y_unit_degree) + 1
    logger.info("X size: %s", x_size)
    logger.info("Y size: %s", y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.info("T size: %s", t_size)
    logger.info("Spatiotemporal units: %s", [t_size, x_size, y_size, num_topic])
    return [t_size, x_size, y_size, num_topic], x_unit_degree, y_unit_degree


def raw_txy_to_index_txy_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, t: str, x, y, x_unit_degree, y_unit_degree):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], x)
    x1y2 = (y, aoi[0])
    x2y2 = (y, x)
    x_index = int((x - aoi[0])//x_unit_degree)
    y_index = int((y - aoi[1])//y_unit_degree)
    # logger.info("X index: %s", x_index)
    # logger.info("Y index: %s", y_index)
    timestart = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')

    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    # For 1h shift:
    # t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=3)

    t_index = int((t - timestart) / datetime.timedelta(minutes=unit_temporal))
    # x_index = int((x - aoi[0])/unit_spatial)
    # y_index = int((y - aoi[1])/unit_spatial)
    return [t_index, x_index, y_index]


def raw_txy_to_index_txy_metric(aoi, timestart, timeend, unit_spatial, unit_temporal, t: str, x, y):
    x1y1 = (aoi[1], aoi[0])
    x2y1 = (aoi[1], x)
    x1y2 = (y, aoi[0])
    x2y2 = (y, x)
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    # logger.debug("X distance: %s meters", x_distance)
    # logger.debug("Y distance: %s meters", y_distance)
    x_index = int(x_distance // unit_spatial)
    y_index = int(y_distance // unit_spatial)
    # logger.info("X size: %s", x_index)
    # logger.info("Y size: %s", y_index)
    timestart = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    t_index = int((t - timestart) / datetime.timedelta(minutes=unit_temporal))
    # x_index = int((x - aoi[0])/unit_spatial)
    # y_index = int((y - aoi[1])/unit_spatial)
    return [t_index, x_index, y_index]


def define_spatiotemporal_unit_degree(aoi, timestart, timeend, unit_spatial, unit_temporal):
    x_start = int(round(aoi[0] / unit_spatial))
    logger.debug(x_start)
    y_start = int(round(aoi[1] / unit_spatial))
    logger.debug(y_start)
    x_end = int(round(aoi[2] / unit_spatial))
    logger.debug(x_end)
    y_end = int(round(aoi[3] / unit_spatial))
    logger.debug(y_end)
    x_size = round(x_end - x_start)
    logger.debug(x_size)
    y_size = round(y_end - y_start)
    logger.debug(y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.debug(t_size)
    return [t_size, x_size, y_size]


def define_spatiotemporal_unit_topic_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, num_topic):
    x_start = int(round(aoi[0] / unit_spatial))
    logger.debug(x_start)
    y_start = int(round(aoi[1] / unit_spatial))
    logger.debug(y_start)
    x_end = int(round(aoi[2] / unit_spatial))
    logger.debug(x_end)
    y_end = int(round(aoi[3] / unit_spatial))
    logger.debug(y_end)
    x_size = round(x_end - x_start)
    logger.debug(x_size)
    y_size = round(y_end - y_start)
    logger.debug(y_size)
    t_start = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t_end = datetime.datetime.strptime(timeend, '%Y-%m-%d %H:%M:%S')
    t_size = round((t_end - t_start) / datetime.timedelta(minutes=unit_temporal))
    logger.debug(t_size)
    # t_size = int(t_size)
    # logger.debug(t_size)
    return [t_size, x_size, y_size, num_topic]


def raw_txy_to_index_txy_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, t: str, x, y):
    timestart = datetime.datetime.strptime(timestart, '%Y-%m-%d %H:%M:%S')
    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    t_index = int((t - timestart) / datetime.timedelta(minutes=unit_temporal))
    x_index = int((x - aoi[0]) / unit_spatial)
    y_index = int((y - aoi[1]) / unit_spatial)
    return [t_index, x_index, y_index]


# def tweets_to_grid(engine_conf, table, st_units, outfile):
#     density = numpy.zeros(st_units)
#     engine = sqlalchemy.create_engine(engine_conf, echo=False)
#     conn = engine.connect()
#     metadata = sqlalchemy.MetaData(engine)
#     table = sqlalchemy.Table(table, metadata, autoload=True, autoload_with=engine)
#     s = sqlalchemy.select([table.c.id_str, table.c.tweeted_at, table.c.y, table.c.x, table.c.words])\
#         .where(sqlalchemy.between(table.c.tweeted_at, timestart, timeend))\
#         .where(sqlalchemy.between(table.c.x, aoi[0], aoi[2]))\
#         .where(sqlalchemy.between(table.c.y, aoi[1], aoi[3]))
#     result = conn.execute(s)
#     for row in result.fetchall():
#         id_str = row['id_str']
#         words = row['words']
#         t = str(row['tweeted_at'])
#         x = float(row['x'])
#         y = float(row['y'])
#         logger.debug(id_str)
#         try:
#             index = raw_txy_to_index_txy_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
#             density[index[0], index[1], index[2]] += 1
#         except IndexError as err:
#             logger.debug(t, err)
#             continue
#     f = open(outfile, 'wb')
#     logger.debug("saving.")
#     pickle.dump(density, f)
#     f.close()
#     result.close()


if __name__ == '__main__':
    engine_conf = s.ENGINE_CONF
    table = s.TABLE_NAME

    timestart = s.TIMESTART
    timestart_text = s.TIMESTART_TEXT
    timeend = s.TIMEEND
    timeend_text = s.TIMEEND_TEXT
    aoi = s.AOI
    unit_temporal = s.UNIT_TEMPORAL
    unit_spatial = s.UNIT_SPATIAL
    num_topic = s.NUM_TOPIC

    gps_dir = s.GPS_DIR
    tweet_counter_file = s.TWEET_COUNTER_FILE

    # # Retrieval parameters for testing
    # timestart = "2013-07-31 12:00:00"
    # timestart_text = "201307031"
    # timeend = "2013-07-31 23:59:59"
    # timeend_text = "20130731"
    # # Area of interest
    # # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
    # aoi = [139.71, 35.65, 139.77, 35.69]  # Greater Tokyo Area
    #
    # conf = configparser.ConfigParser()
    # conf.read('../config.cfg')
    # remote_db = {
    #     "host": conf.get('RDS', 'host'),
    #     "user": conf.get('RDS', 'user'),
    #     "passwd": conf.get('RDS', 'passwd'),
    #     "db_name": conf.get('RDS', 'db_name'),
    # }
    # engine_conf = "mysql+pymysql://" + remote_db["user"] + ":" + remote_db["passwd"] + "@" + remote_db["host"] + "/" + remote_db["db_name"] + "?charset=utf8mb4"
    # table = "social_activity_201307"

    # # Retrieval parameters
    # timestart = "2013-07-01 00:00:00"
    # timestart_text = "20130701"
    # timeend = "2013-07-31 23:59:59"
    # timeend_text = "20130731"
    # # Area of interest
    # # aoi = [122.933198,24.045416,153.986939,45.522785] # Japan
    # aoi = [138.72, 34.9, 140.87, 36.28]  # Greater Tokyo Area

    unit_temporal = 60  # in minutes
    unit_spatial = 1000  # in degrees
    example_x = 138.750000
    example_y = 34.91000
    example_t = "2013-07-03 00:00:00"

    # st_units = define_spatiotemporal_unit_degree(aoi, timestart, timeend, unit_spatial, unit_temporal)
    st_units, x_unit_degree, y_unit_degree = define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal)
    logger.debug(st_units)
    example_index = raw_txy_to_index_txy_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, example_t, example_x, example_y, x_unit_degree, y_unit_degree)
    logger.debug(example_index)
    # tweets_to_grid(engine_conf, table, st_units, tweet_counter_file)
    # logger.debug("Done.")
