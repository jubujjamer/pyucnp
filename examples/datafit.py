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

t, fsignal, weights, fit_dict = dynamics.ucfit(tdata, ydata, fitlist, abserr=5E-5, curve='E')

for par in fitlist:
    print('For %s -> adjusted value: %.4f with weight: %.2f' % (par, fit_dict[par], weights[par]) )

#with open('ucsystem.dat', 'w') as f:
    # Print & save the solution.
#    w1[0], w1[1], w1[2], w1[3], w1[4])
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[14, 4])
ax.plot(t, fsignal, label='Verde (Er, 6)')
ax.plot(tdata, ydata)
#ax.set_ylim([0, 1]);
ax.legend()
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud')
plt.show()
