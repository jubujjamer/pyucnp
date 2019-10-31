# -*- coding: utf-8 -*-
"""
file data.py
author: Juan Marco Bujjamer

Data fitting for UCNP curves.
"""
import os
import numpy as np
import collections
import matplotlib.pylab as plt
import scipy
import pandas as pd
from . import fitting as df
from pathlib import Path
import yaml
import logging
from pathlib import Path

DATA_FOLDER = '/home/juan/pCloudDrive/doctorado/UCNP/meds/'
SPEC_DEFAULT = 'sample1.txt'
SPEC_YAML = 'data_info.yaml'
DATA_DEFAULT = 'sample_1.yaml'
er_ion_path = Path('~/git/pyucnp/data/erbium_carnall.csv').expanduser()
er_transitions_path = Path('~/git/pyucnp/data/erbium_transitions_carnall.csv').expanduser()

def load_ion_states(ion_name='erbium'):
    if ion_name == 'erbium':
        er_ion = pd.read_csv(er_ion_path, header=2,
                          names=['observed', 'calculated',
                          'o-c', 'state', 'j', 'mj'], delimiter='\s+')
    return er_ion

def load_ion_transitions(ion_name='erbium'):
    if ion_name == 'erbium':
        er_ion = pd.read_csv(er_transitions_path, header=2,
                          names=['from', 'to',
                          'ajj', 'kr', 'beta'], delimiter='\s+')
    return er_ion

def load_data(daystr, config_file=None):
    """ Loads data from de measurements folder.
    Parameters:
    -----------
    daystr          string
                    Measurement day in the yyyy-mm-dd format.
    config_file     string
                    Config file as a yaml file, tipically stored in the same
                    measurement folder.

    Returns
    -------
    cfg             named tuple
                    Named tuple with the needed variables.
    """
    basedir = Path(DATA_FOLDER, daystr)
    if not config_file:
        config_file = DATA_DEFAULT
    yaml_fin = basedir / config_file
    yaml_fin = yaml_fin.with_suffix('.yaml')
    logging.info('Loading cfg from file %s' % yaml_fin)
    try:
        config_dict = yaml.load(open(yaml_fin, 'r'))
    except:
        raise FileNotFoundError('File %s not found nor valid.' % config_file)
    config = collections.namedtuple('config', config_dict.keys())
    cfg = config(*config_dict.values())
    return cfg


<<<<<<< HEAD
def load_spectrum(daystr, nmeas, sample=None):
=======
def load_felix_spectrum(spec_file, nmeas):
    """ Get Felix PTI spectrum form csv files.
    Parameters:
    daystr:     str
    nmeas

    wavelength_list
    """
    columns = list()

    HEADER = 4
    for i in range(132):
        columns.append(str(i))
    df = pd.read_csv(spec_file, delimiter='\t', names=columns,
                        header=None)
    x = df[str((nmeas-1)*2)][HEADER:]
    y = df[str((nmeas-1)*2+1)][HEADER:]
    x = np.array([float(x_i) for x_i in x])
    y = np.array([float(y_i) for y_i in y])
    try:
        first_nan = np.where(np.isnan(x))[0][0]
    except:
        first_nan = -1
    return x[:first_nan], y[:first_nan]

def load_spectrum(daystr, nmeas, fname=None):
>>>>>>> b9c33793c1bc88e007c4c6063101171787aa5236
    """ Get Felix PTI spectrum form csv files.
    Parameters:
    daystr:     str
    nmeas

    wavelength_list
    """
    basedir = Path(DATA_FOLDER, daystr)
    if not sample:
        sample = SPEC_DEFAULT
    data_fin = basedir / sample
    data_fin = data_fin.with_suffix('.txt')
    logging.info('Loading data from file %s' % data_fin)
    columns = list()
    for i in range(132):
        columns.append(str(i))
    table = pd.read_csv(data_fin, header=None, delimiter='\t', names=columns)
    head = 4
    x = table[str((nmeas-1)*2)][head:]
    y = table[str((nmeas-1)*2+1)][head:]
    x = np.array([float(x_i) for x_i in x])
    y = np.array([float(y_i) for y_i in y])
    try:
        first_nan = np.where(np.isnan(x))[0][0]
    except:
        first_nan = -1
    return x[:first_nan], y[:first_nan]

def save_pickled(daystr, sample, spectra):
    """ Save SpectralData object as pickle file.

    """
    import pickle
    pickle_file = Path(DATA_FOLDER, daystr, sample)
    pickle_file = pickle_file.with_suffix('.sp')
    logging.info('Saving pickle file %s.' % pickle_file)
    with open(pickle_file, 'wb') as outfile:
        pickle.dump(spectra, outfile)

def load_pickled(daystr, sample):
    """ Load SpectralData object from pickle file.

    """
    import pickle
    pickle_file = Path(DATA_FOLDER, daystr, sample)
    pickle_file = pickle_file.with_suffix('.sp')
    logging.info('Loading pickle file %s.' % pickle_file)
    with open(pickle_file, 'rb') as outfile:
        sdata = pickle.load(outfile)
    return sdata

def UCNPEmission(object):
    def __init__(self):
        __all__ = ['__init__']

def load_idecay(daystr, filename, nbins=60, ndata=-1, TS=6.4E-8):
    basedir = os.path.join(DATA_FOLDER, daystr)

    data_fin = os.path.join(basedir, filename + '.npy')
    text_fin = os.path.join(basedir, filename + '.txt')
    # with open(text_fin, 'r') as textfile:
    #     title = textfile.readlines()[3]
    title = filename
    time_data = np.load(data_fin)*TS
    hist, edges = np.histogram(time_data, bins=nbins, density=False)
    times = edges[0:-1]
    y = hist - np.mean(hist[-nbins//20: -1])
    ydata = y/np.max(y)
    return times, ydata

def load_mean_times(daystr, filename, nbins=60, ndata=-1, TS=6.4E-8):
    basedir = os.path.join(DATA_FOLDER, daystr)

    data_fin = os.path.join(basedir, filename + '.npy')
    text_fin = os.path.join(basedir, filename + '.txt')
    title = filename
    time_data = np.load(data_fin)*TS
    return np.mean(time_data)


def get_timesets(daystr, nbins=2500, model='mixture', filtering=True,
                 ndata=-1, datafile=DATA_DEFAULT):
    """ Load time decay data.

    Parameters
    ----------
    daystr : string
        Name of the stored data folder (e.g. 2017-09-22).
    nbins : int
        Number of bins points used in the time data.
    model : string
        Model for the data fitting. Supported models in fitting.fit_exponential().
    filtering : bool
        Should the data be previously filtered or not.
    ndata : int
        Number of points in each time series.
    datafile : string
        yaml file with extra information on the measurements.

    Returns
    -------
    list
        Description of returned object.

    """
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, datafile)
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            datasets = yaml_data['datasets']
        except yaml.YAMLError as exc:
            print(exc)

    datasets_list = list()
    for key, filenames in iter(datasets.items()):
        datasets_list.append(get_time_data(daystr, nbins=nbins, model=model,
                                           filtering=filtering, ndata=ndata,
                                           filenames=filenames, datafile=datafile))
    return datasets_list



def load_timedata(daystr, filenames, nbins=60, ndata=-1):

    TS = 6.4E-8 # TODO: get this from the config file
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


def get_time_data(daystr, nbins=2500, model='mixture', filtering=True, ndata=-1,
                  filenames=None, datafile=DATA_DEFAULT):
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, datafile)
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
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



def load_multiple_spectra(daystr, datafile=DATA_DEFAULT):
    import yaml
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, datafile)
    spec_collection = []
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            spectrum_names = yaml_data['spectrum_names']
            spectra_to_plot = yaml_data['spectra_to_plot']
        except yaml.YAMLError as exc:
            print(exc)
    for m in spectra_to_plot:
        x, y = load_spectrum(daystr, m)
        y = y - np.mean(y[-10:-1])
        spec_collection.append( (np.asarray(x), np.asarray(y)) )
        # if m == 2:
        #     norm = scipy.integrate.simps(y)
        # pvals.append(scipy.integrate.simps(y)/norm)
    return spec_collection



def get_powers(daystr, datafile=None):
    basedir = os.path.join(DATA_FOLDER, daystr)
    yaml_fin = os.path.join(basedir, datafile)
    with open(yaml_fin, 'r') as stream:
        try:
            yaml_data = yaml.load(stream)
            lpowers = yaml_data['laser_vpp_list']
        except yaml.YAMLError as exc:
            print(exc)
    return lpowers


def integrate_power(x, y, center=540, bwidth=10):
    ci = np.where(x==center)[0][0]  # Center index
    intp = scipy.integrate.simps(y[ci-bwidth//2:ci+bwidth//2])
    return intp


def get_timepars(daystr=None, nbins=1200, model='double_neg', filtering=False,
                 ndata=300, dsnumber=0, datafile=DATA_DEFAULT):
    datasets_list = get_timesets(daystr, nbins=nbins, model=model,
                                 filtering=filtering, ndata=ndata,
                                 datafile=datafile)
    dataset = datasets_list[dsnumber]
    a1_list, a2_list, ka_list, kUC_list = list(), list(), list(), list()
    for tdata, ydata, results in dataset:
        a1_list.append(results.params['a1'].value)
        a2_list.append(results.params['a2'].value)
        ka_list.append(results.params['ka'].value)
        kUC_list.append(results.params['kUC'].value)
    return np.array(a1_list), np.array(a2_list), np.array(ka_list), np.array(kUC_list)
