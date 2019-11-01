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
import numpy as np
import pandas as pd
from . import fitting as df
from pathlib import Path
import yaml
import logging
import h5py

DATA_PATH= Path('~/pCloudDrive/doctorado/UCNP/meds/').expanduser()
DATA_FOLDER = '/home/juan/pCloudDrive/doctorado/UCNP/meds/'
SPEC_DEFAULT = 'sample_1.txt'
SPEC_YAML = 'data_info.yaml'
CONFIG_DEFAULT = 'sample_1.yaml'
er_ion_path = Path('~/git/pyucnp/data/erbium_carnall.csv').expanduser()
er_transitions_path = Path('~/git/pyucnp/data/erbium_transitions_carnall.csv').expanduser()


def get_basedir(daystr):
    """ Returns a path to the where the data is stored
    """
    return DATA_PATH/daystr

def load_ion_states(ion_name='erbium'):
    """ Load ion energy levels data table.

    From Carnall et al.
    """
    if ion_name == 'erbium':
        er_ion = pd.read_csv(er_ion_path, header=2,
                          names=['observed', 'calculated',
                          'o-c', 'state', 'j', 'mj'], delimiter='\s+')
    return er_ion

def load_ion_transitions(ion_name='erbium'):
    """ Load transition probabilities for a trivalent ion.
    """
    if ion_name == 'erbium':
        er_ion = pd.read_csv(er_transitions_path, header=2,
                          names=['from', 'to',
                          'ajj', 'kr', 'beta'], delimiter='\s+')
    return er_ion

# def get_decay_names(daystr,):
#     """ Gets the decay curves names and
#
#     Parameters:
#     -----------
#     daystr          string
#                     Measurement day in the yyyy-mm-dd format.
#
#
#     Returns
#     -------
#     cfg             named tuple
#                     Named tuple with the needed variables.
#     """
#     basedir = get_basedir(daystr)
#     formats = ['hdf', 'npy']
#     format = None
#     for fmt in formats:
#         imfiles = basedir.glob(f'**/*.{fmt}')
#         files = [m.name for m in imfiles]
#         if len(files):
#             format = fmt
#             return files, format

def load_cfg(daystr, config_file=None):
    """ Loads data from the measurements folder.
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
    basedir = get_basedir(daystr)
    if not config_file:
        config_file = CONFIG_DEFAULT
    logging.info('Loading data from file %s' % config_file)
    yaml_fin = os.path.join(basedir, config_file)
    try:
        config_dict = yaml.load(open(yaml_fin, 'r'), Loader=yaml.FullLoader)
    except:
        raise FileNotFoundError('File %s not found nor valid.' % config_file)
    config = collections.namedtuple('config', config_dict.keys())
    cfg = config(*config_dict.values())
    return cfg

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
    """ Get Felix PTI spectrum form csv files.
    Parameters:
    daystr:     str
    nmeas

    wavelength_list
    """
    basedir = os.path.join(DATA_FOLDER, daystr)
    if not fname:
        data_fin = os.path.join(basedir, SPEC_DEFAULT)
    else:
        data_fin = os.path.join(basedir, fname)

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

def save_pickled(daystr, filename, spectra):
    """ Save spectra object as pickle file.

    """
    import pickle
    basedir = os.path.join(DATA_FOLDER, daystr)
    pickle_file = os.path.join(basedir, filename+'.sp')
    with open(pickle_file, 'wb') as outfile:
        pickle.dump(spectra, outfile)

def load_pickled(daystr, filename):
    import pickle
    basedir = os.path.join(DATA_FOLDER, daystr)
    pickle_file = os.path.join(basedir, filename)
    print(pickle_file)
    with open(pickle_file, 'rb') as outfile:
        sdata = pickle.load(outfile)
    return sdata
# def UCNPEmission(object):
#     def __init__(self):
#         __all__ = ['__init__']

def load_idecay(daystr, filename, nbins=60, ndata=-1, TS=6.4E-8):
    """ Load time resolved data.
    """
    basedir = DATA_PATH/daystr
    if format == 'npy':
        data_fin = (basedir/filename).with_suffix('npy')
        text_fin = (basedir/filename).with_suffix('txt')
        title = filename
        time_data = np.load(data_fin)*TS
        hist, edges = np.histogram(time_data, bins=nbins, density=False)
        times = edges[0:-1]
        y = hist - np.mean(hist[-nbins//20: -1])
        ydata = y/np.max(y)
    return times, idata

def load_hdf_idecay(daystr, index, wlen, nbins=80, ndata=-1, TS=6.4E-8):
    """ Load idecay in hdf format

    Name of the files are biult like 'wlen-YYYYMMDD-HHMMSS.hdf'
    (e.g. 410-20190207-183552.hdf)

    HDF files format:
        counts
            array of photon counts
        time
            array of arrival times
        idata
            duty_cycle: 20 <class 'str'>
            laser_power: 3 V <class 'str'>
            optical_filter: ODE 0 <class 'str'>
            wavelength: 410 <class 'numpy.int32'>
    """
    basedir = DATA_PATH/daystr
    cfg = load_cfg(daystr)
    hdfname = [name for name in cfg.idecay_datasets[index] if str(wlen) in name]
    data_fin = (basedir/hdfname[0]).with_suffix('.hdf')
    with h5py.File(data_fin, 'r') as hdf:
        time_data = hdf['counts'][()]*TS
        hist, edges = np.histogram(time_data, bins=nbins, density=False)
        time = hdf['time'][()]
        ydata = hdf['idata'][()]
        power = hdf['idata'].attrs['laser_power']
    return time, ydata, power


def load_mean_times(daystr, filename, nbins=60, ndata=-1, TS=6.4E-8):
    """
    """
    basedir = os.path.join(DATA_FOLDER, daystr)

    data_fin = os.path.join(basedir, filename + '.npy')
    text_fin = os.path.join(basedir, filename + '.txt')
    title = filename
    time_data = np.load(data_fin)*TS
    return np.mean(time_data)

# def get_timesets(daystr, nbins=2500, model='mixture', filtering=True,
#                  ndata=-1, datafile=CONFIG_DEFAULT):
#     """ Load time decay data.
#
#     Parameters
#     ----------
#     daystr : string
#         Name of the stored data folder (e.g. 2017-09-22).
#     nbins : int
#         Number of bins points used in the time data.
#     model : string
#         Model for the data fitting. Supported models in fitting.fit_exponential().
#     filtering : bool
#         Should the data be previously filtered or not.
#     ndata : int
#         Number of points in each time series.
#     datafile : string
#         yaml file with extra information on the measurements.
#
#     Returns
#     -------
#     list
#         Description of returned object.
#
#     """
#     import yaml
#     basedir = os.path.join(DATA_FOLDER, daystr)
#     yaml_fin = os.path.join(basedir, datafile)
#     with open(yaml_fin, 'r') as stream:
#         try:
#             yaml_data = yaml.load(stream)
#             datasets = yaml_data['datasets']
#         except yaml.YAMLError as exc:
#             print(exc)
#
#     datasets_list = list()
#     for key, filenames in iter(datasets.items()):
#         datasets_list.append(get_time_data(daystr, nbins=nbins, model=model,
#                                            filtering=filtering, ndata=ndata,
#                                            filenames=filenames, datafile=datafile))
#     return datasets_list

# def load_timedata(daystr, filenames, nbins=60, ndata=-1):
#     TS = 6.4E-8 # TODO: get this from the config file
#     basedir = os.path.join(DATA_FOLDER, daystr)
#     data_list = list()
#     for filename in filenames:
#         data_fin = os.path.join(basedir, filename + '.npy')
#         text_fin = os.path.join(basedir, filename + '.txt')
#         textfile = open(text_fin, 'r')
#         title = textfile.readlines()[3]
#         title = filename
#         textfile.close()
#         time_data = np.load(data_fin)*TS
#         NACQ = len(time_data)
#         hist, edges = np.histogram(time_data, bins=nbins, density=False)
#         times = edges[0:-1]
#         y = hist - np.mean(hist[-nbins//20: -1])
#         ydata = y/np.max(y)
#         data_list.append((times[:ndata], ydata[:ndata], title))
#     return data_list

# def get_time_data(daystr, nbins=2500, model='mixture', filtering=True, ndata=-1,
#                   filenames=None, datafile=CONFIG_DEFAULT):
#     import yaml
#     basedir = os.path.join(DATA_FOLDER, daystr)
#     yaml_fin = os.path.join(basedir, datafile)
#     with open(yaml_fin, 'r') as stream:
#         try:
#             yaml_data = yaml.load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     b, a = scipy.signal.butter(2, 0.3)# Para el filtrado (de ser necesario)
#     data_list = list()
#     # Hago los ajustes y guardo los parámetros ajustados
#     for tdata, ydata, title in load_timedata(daystr, filenames,
#                                              nbins=nbins, ndata=ndata):
#         if filtering:
#             ydata = scipy.signal.filtfilt(b, a, ydata)
#         result = df.robust_fit(tdata, ydata, model=model)
#         data_list.append((tdata, ydata, result))
#     return data_list

# def integrate_power(x, y, center=540, bwidth=10):
#     ci = np.where(x==center)[0][0]  # Center index
#     intp = scipy.integrate.simps(y[ci-bwidth//2:ci+bwidth//2])
#     return intp

# def get_timepars(daystr=None, nbins=1200, model='double_neg', filtering=False,
#                  ndata=300, dsnumber=0, datafile=CONFIG_DEFAULT):
#     datasets_list = get_timesets(daystr, nbins=nbins, model=model,
#                                  filtering=filtering, ndata=ndata,
#                                  datafile=datafile)
#     dataset = datasets_list[dsnumber]
#     a1_list, a2_list, ka_list, kUC_list = list(), list(), list(), list()
#     for tdata, ydata, results in dataset:
#         a1_list.append(results.params['a1'].value)
#         a2_list.append(results.params['a2'].value)
#         ka_list.append(results.params['ka'].value)
#         kUC_list.append(results.params['kUC'].value)
#     return np.array(a1_list), np.array(a2_list), np.array(ka_list), np.array(kUC_list)
