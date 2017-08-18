# -*- coding: utf-8 -*-
"""
file data.py
author: Juan Marco Bujjamer

Data fitting for UCNP curves.
"""
import os
import numpy as np
import matplotlib.pylab as plt
import scipy
import pandas as pd

DATA_FOLDER = '/home/lec/pCloudDrive/doctorado/UCNP/meds/'

def load_timedata(daystr, filenames, nbins=60, ndata=-1):
    TS = 6.4E-8
    basedir = os.path.join(DATA_FOLDER, daystr)
    data_list = list()
    for filename in filenames:
        data_fin = os.path.join(basedir, filename + '.npy')
        text_fin = os.path.join(basedir, filename + '.txt')
        textfile = open(text_fin, 'r')
        title = textfile.readlines()[3]
        title = filename
        textfile.close()
        time_data = np.load(data_fin)*TS
        NACQ = len(time_data)
        hist, edges = np.histogram(time_data, bins=nbins, density=False)
        times = edges[0:-1]
        y = hist - np.mean(hist[-nbins//20: -1])
        ydata = y/np.mean(y[0:8])
        data_list.append((times[:ndata], ydata[:ndata], title))
    return data_list

def print_datainfo(daystr, filenames):
    basedir = os.path.join(DATA_FOLDER, daystr)
    data_list = list()
    for filename in filenames:
        data_fin = os.path.join(basedir, filename + '.npy')
        text_fin = os.path.join(basedir, filename + '.txt')
        textfile = open(text_fin, 'r')
        datainfo = textfile.readlines()
        textfile.close()
        print(filename)
        print(datainfo)
    return


def load_spectrum(daystr, filename, meas_n):
    """ Get Felix PTI spectrum.
    """
    basedir = os.path.join(DATA_FOLDER, daystr)
    data_fin = os.path.join(basedir, filename)
    columns = list()
    for i in range(132):
        columns.append(str(i))
    df = pd.read_csv(data_fin, header=None, delimiter='\t', names=columns)
    head = 4
    x = df[str((meas_n-1)*2)][head:]
    y = df[str((meas_n-1)*2+1)][head:]
    x = [float(x_i) for x_i in x]
    y = [float(y_i) for y_i in y]
    try:
        first_nan = np.where(np.isnan(x))[0][0]
    except:
        first_nan = -1
    return x[:first_nan], y[:first_nan]
