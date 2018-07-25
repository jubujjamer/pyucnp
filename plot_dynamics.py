# -*- coding: utf-8 -*-
"""
file datafit.py
author: Juan Marco Bujjamer

Data fitting for UCNP curves.
"""
import os
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.stats import chisquare
import matplotlib.cm as cm


from pyucnp import dynamics
# The data file
DATA_FOLDER = '/home/lec/pCloudDrive/doctorado/UCNP/meds/'
FILE = '2017-08-02/HEPES_01.npy'
fitlist = ['WA',  'WB', 'WC', 'WE',
           'WCB', 'WDB', 'WDC', 'WEB', 'WEC', 'WED', 'WD',
           'A0', 'B0', 'C0', 'D0', 'E0', 'R0', 'S0']

infile = os.path.join(DATA_FOLDER, FILE)

TS = 6.4E-8
nbins = 250
time_data = np.load(infile)*TS
NACQ = len(time_data)
hist, edges = np.histogram(time_data, bins=nbins, density=False)

tdata = edges[0:-1]
ydata = (hist - np.mean(hist[-nbins//4:-1]))
ydata /= ydata[0]
t, wsol = dynamics.simulate_simple_input(np.linspace(-2, 1, 10000), ydata, state='N2')

fig, axes = plt.subplots(ncols=1, nrows=3, figsize=[12, 12])
for i, label in enumerate(['N0', 'N1', 'N2']):
    axes[i].plot(t, wsol[i], label=label, color=cm.PuOr(i*20+5))
# ax.plot(tdata, ydata)
for ax in axes:
    ax.set_xlim([-.1, 1]);
    ax.legend()
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
plt.show()
