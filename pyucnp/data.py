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
import fitting as df

DATA_FOLDER = '/home/lec/pCloudDrive/doctorado/UCNP/meds/'
SPEC_DEFAULT = 'specs.txt'
SPEC_YAML = 'data_info.yaml'

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
        ydata = y/np.max(y)
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

def load_multiple_spectra(daystr):
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, SPEC_YAML)
    spec_collection = []
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            spectrum_names = yaml_data['spectrum_names']
            spectra_to_plot = yaml_data['spectra_to_plot']
        except yaml.YAMLError as exc:
            print(exc)
    for m in spectra_to_plot:
        x, y = load_spectrum(daystr, SPEC_DEFAULT, m)
        y = y - np.mean(y[-10:-1])
        spec_collection.append( (np.asarray(x), np.asarray(y)) )
        # if m == 2:
        #     norm = scipy.integrate.simps(y)
        # pvals.append(scipy.integrate.simps(y)/norm)
    return spec_collection


def get_time_data(daystr, nbins=2500, model='mixture', filtering=True, ndata=-1, filenames=None):
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, SPEC_YAML)
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            lpowers = yaml_data['laser_vpp_list']
        except yaml.YAMLError as exc:
            print(exc)
    b, a = scipy.signal.butter(2, 0.3)# Para el filtrado (de ser necesario)
    data_list = list()
    # Hago los ajustes y guardo los parámetros ajustados
    for tdata, ydata, title in load_timedata(daystr, filenames,
                                             nbins=nbins, ndata=ndata):
        if filtering:
            ydata = scipy.signal.filtfilt(b, a, ydata)
        result = df.robust_fit(tdata, ydata, model=model)
        data_list.append((tdata, ydata, result))
    return data_list


def get_timesets(daystr, nbins=2500, model='mixture', filtering=True, ndata=-1):
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, SPEC_YAML)
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            datasets = yaml_data['datasets']
        except yaml.YAMLError as exc:
            print(exc)

    datasets_list = list()
    for key, filenames in datasets.iteritems():
        datasets_list.append(get_time_data(daystr, nbins=nbins, model=model,
                                           filtering=filtering, ndata=ndata,
                                           filenames=filenames))
    return datasets_list
