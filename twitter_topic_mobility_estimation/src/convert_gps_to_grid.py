__author__ = 'koitaroh'

# convert_gps_to_grid.py
# Last Update: 2016-08-14
# Author: Satoshi Miyazawa
# koitaroh@gmail.com
# Convert GPS trajectories to population density array p[t, x, y]

import csv
import datetime
import logging
import os
import pickle
from logging import handlers

import geopy
import numpy
import pandas

import convert_points_to_grid
import settings as s

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


def count_user(gps_dir):
    user = set()
    for root, dirs, files in os.walk(gps_dir):
        filelist = files[0:100]
        for fn in filelist:
            logger.info(fn)
            if fn[0] != '.':
                with open(root + os.sep + fn) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        user.add(row[0])
            logger.info(len(user))
    logger.info("Final count: %s", len(user))


def gps_to_grid_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, gps_dir, gps_counter_file):
    density = numpy.zeros(st_units, dtype=numpy.int)
    for root, dirs, files in os.walk(gps_dir):
        filelist = files[0:100]
        # logger.debug(filelist)
        for fn in filelist:
            print(fn)
            if fn[0] != '.':
                with open(root + os.sep + fn) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        # print(row)
                        if len(row) == 6:
                            t = str(row[1])
                            y = float(row[2])
                            x = float(row[3])
                            if (aoi[0] <= x <=aoi[2]) and (aoi[1] <= y <= aoi[3]):
                                try:
                                    index = convert_points_to_grid.raw_txy_to_index_txy_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
                                    # index = raw_txy_to_index_txy_degree(timestart, timeend, aoi, unit_temporal, unit_spatial, t, x, y)
                                    # logger.debug(index)
                                    density[index[0], index[1], index[2]] += 1
                                except IndexError as err:
                                    logger.debug(err)
                                    continue
    f = open(gps_counter_file, 'wb')
    logger.debug("saving.")
    pickle.dump(density, f)
    f.close()


def raw_txy_to_index_txy_metric_pandas(args):
    uid = args[0]
    t = args[1]
    x = args[3]
    y = args[2]
    # logger.debug("t: %s", t)
    # logger.debug("x: %s", x)
    # logger.debug("y: %s", y)
    x1y1 = (AOI[1], AOI[0])
    x2y1 = (AOI[1], x)
    x1y2 = (y, AOI[0])
    x2y2 = (y, x)
    x_distance = geopy.distance.vincenty(x1y1, x2y1).meters
    y_distance = geopy.distance.vincenty(x1y1, x1y2).meters
    # logger.debug("X distance: %s meters", x_distance)
    # logger.debug("Y distance: %s meters", y_distance)
    x_index = int(x_distance // unit_spatial)
    y_index = int(y_distance // unit_spatial)
    # logger.info("X size: %s", x_index)
    # logger.info("Y size: %s", y_index)
    timestart = datetime.datetime.strptime(TIMESTART, '%Y-%m-%d %H:%M:%S')
    # t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    t_index = int((t - timestart)/datetime.timedelta(minutes=unit_temporal))
    # x_index = int((x - aoi[0])/unit_spatial)
    # y_index = int((y - aoi[1])/unit_spatial)
    return pandas.Series({'uid': uid, 't_index': t_index, 'x_index': x_index, 'y_index': y_index})


def gps_to_grid_metric_people_new(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, x_unit_degree, y_unit_degree, gps_dir, gps_counter_file):
    density = numpy.zeros(st_units, dtype=numpy.int)
    os.makedirs(MODELS_DIR + "/GPS/")
    for root, dirs, files in os.walk(gps_dir):
        files.sort()
        filelist = files[0:120]
        # logger.debug(filelist)
        for fn in filelist:
            logger.info(fn)
            if fn[0] != '.':
                with open(root + os.sep + fn) as csvfile:
                    reader = csv.reader(csvfile)
                    with open(MODELS_DIR + "/GPS/" + 'ex_' + fn, 'w', newline='') as csvfile_new:
                        writer = csv.writer(csvfile_new, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for row in reader:
                            if len(row) == 6:
                                uid = str(row[0])
                                t = str(row[1])
                                y = float(row[2])
                                x = float(row[3])
                                if (aoi[0] <= x <=aoi[2]) and (aoi[1] <= y <= aoi[3]):
                                    try:
                                        index = convert_points_to_grid.raw_txy_to_index_txy_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y, x_unit_degree, y_unit_degree)

                                        if uid and index[0] and index[1] and index[2] is not None:
                                            writer.writerow([uid, index[0], index[1], index[2]])
                                    except IndexError as err:
                                        # logger.debug("Original coordinates: t: %s, x: %s, y: %s", t, x, y)
                                        # logger.debug(err)
                                        continue
                        gps_df = pandas.read_csv(MODELS_DIR + "/GPS/" + 'ex_' + fn, header=None, names=['uid', 't_index', 'x_index', 'y_index'])
                        # gps_df_1 = gps_df[:10000]
                        logger.debug("Original dataframe length: %s", gps_df.__len__())
                        gps_df = gps_df.drop_duplicates()
                        logger.debug("Duplicates dropped length: %s", gps_df.__len__())
                        for index, row in gps_df.iterrows():
                            try:
                                # index = [int(row['t_index']), int(row['x_index']), int(row['y_index'])]
                                # logger.debug("Adding %s, %s, %s", int(row['t_index']), int(row['x_index']), int(row['y_index']))
                                density[int(row['t_index']), int(row['x_index']), int(row['y_index'])] += 1
                            except IndexError as err:
                                # logger.debug(err)
                                continue
                            except ValueError as err:
                                # logger.debug(err)
                                continue
    f = open(gps_counter_file, 'wb')
    logger.debug("saving.")
    pickle.dump(density, f)
    f.close()
    return None

def gps_to_grid_metric_people(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, gps_dir, gps_counter_file):
    density = numpy.zeros(st_units, dtype=numpy.int)
    os.makedirs(MODELS_DIR + "/GPS/")
    for root, dirs, files in os.walk(gps_dir):
        files.sort()
        filelist = files[0:120]
        # logger.debug(filelist)
        for fn in filelist:
            logger.info(fn)
            if fn[0] != '.':
                with open(root + os.sep + fn) as csvfile:
                    reader = csv.reader(csvfile)
                    with open(MODELS_DIR + "/GPS/" + 'ex_' + fn, 'w', newline='') as csvfile_new:
                        writer = csv.writer(csvfile_new, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        for row in reader:
                            if len(row) == 6:
                                uid = str(row[0])
                                t = str(row[1])
                                y = float(row[2])
                                x = float(row[3])
                                if (aoi[0] <= x <=aoi[2]) and (aoi[1] <= y <= aoi[3]):
                                    try:
                                        index = convert_points_to_grid.raw_txy_to_index_txy_metric(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
                                        if uid and index[0] and index[1] and index[2] is not None:
                                            writer.writerow([uid, index[0], index[1], index[2]])
                                    except IndexError as err:
                                        # logger.debug("Original coordinates: t: %s, x: %s, y: %s", t, x, y)
                                        # logger.debug(err)
                                        continue
                        gps_df = pandas.read_csv(MODELS_DIR + "/GPS/" + 'ex_' + fn, header=None, names=['uid', 't_index', 'x_index', 'y_index'])
                        # gps_df_1 = gps_df[:10000]
                        logger.debug("Original dataframe length: %s", gps_df.__len__())
                        gps_df = gps_df.drop_duplicates()
                        logger.debug("Duplicates dropped length: %s", gps_df.__len__())
                        for index, row in gps_df.iterrows():
                            try:
                                # index = [int(row['t_index']), int(row['x_index']), int(row['y_index'])]
                                # logger.debug("Adding %s, %s, %s", int(row['t_index']), int(row['x_index']), int(row['y_index']))
                                density[int(row['t_index']), int(row['x_index']), int(row['y_index'])] += 1
                            except IndexError as err:
                                # logger.debug(err)
                                continue
                            except ValueError as err:
                                # logger.debug(err)
                                continue
    f = open(gps_counter_file, 'wb')
    logger.debug("saving.")
    pickle.dump(density, f)
    f.close()
    return None



def gps_to_grid_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, gps_dir, gps_counter_file):
    density = numpy.zeros(st_units, dtype=numpy.int)
    for root, dirs, files in os.walk(gps_dir):
        filelist = files[0:100]
        # logger.debug(filelist)
        for fn in filelist:
            print(fn)
            if fn[0] != '.':
                with open(root + os.sep + fn) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        # print(row)
                        if len(row) == 6:
                            t = str(row[1])
                            y = float(row[2])
                            x = float(row[3])
                            if (aoi[0] <= x <=aoi[2]) and (aoi[1] <= y <= aoi[3]):
                                try:
                                    index = convert_points_to_grid.raw_txy_to_index_txy_degree(aoi, timestart, timeend, unit_spatial, unit_temporal, t, x, y)
                                    # index = raw_txy_to_index_txy_degree(timestart, timeend, aoi, unit_temporal, unit_spatial, t, x, y)
                                    # logger.debug(index)
                                    density[index[0], index[1], index[2]] += 1
                                except IndexError as err:
                                    logger.debug(err)
                                    continue
    f = open(gps_counter_file, 'wb')
    logger.debug("saving.")
    pickle.dump(density, f)
    f.close()


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


if __name__ == '__main__':
    example_x = 138.750000
    example_y = 34.91000
    example_t = "2013-07-03 00:00:00"
    # count_user(gps_dir)

    # st_units = define_spatiotemporal_unit_degree(timestart, timeend, aoi, unit_temporal, unit_spatial)
    # st_units = convert_points_to_grid.define_spatiotemporal_unit_metric(AOI, TIMESTART, timeend, unit_spatial, unit_temporal)
    # example_index = raw_txy_to_index_txy_degree(timestart, timeend, aoi, unit_temporal, unit_spatial, example_t, example_x, example_y)
    # logger.debug(example_index)
    st_units, x_unit_degree, y_unit_degree = convert_points_to_grid.define_spatiotemporal_unit_metric_new(aoi, timestart, timeend, unit_spatial, unit_temporal)
    # Change function in gps_to_grid_metric depending on metric or degree
    # gps_to_grid_metric(AOI, TIMESTART, timeend, unit_spatial, unit_temporal, st_units, gps_dir, gps_counter_file)
    gps_to_grid_metric_people_new(aoi, timestart, timeend, unit_spatial, unit_temporal, st_units, x_unit_degree, y_unit_degree, gps_dir, gps_counter_file)

