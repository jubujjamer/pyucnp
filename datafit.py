# -*- coding: utf-8 -*-
"""
file datafit.py
author: Juan Marco Bujjamer

Data fitting for UCNP curves.
"""
import os
import numpy as np
import matplotlib.pylab as plt

from pyucnp import dynamics
# The data file
DATA_FOLDER = '/home/lec/pCloudDrive/doctorado/UCNP/meds/'
FILE = '2017-08-02/HEPES_01.npy'

infile = os.path.join(DATA_FOLDER, FILE)

TS = 6.4E-8
nbins = 250
time_data = np.load(infile)*TS
NACQ = len(time_data)
hist, edges = np.histogram(time_data, bins=nbins, density=False)

tdata = edges[0:-1]
ydata = (hist - np.mean(hist[-nbins//4 : -1]))
ydata /= ydata[0]

t, fsignal, delta_cost_list, pfit= dynamics.ucfit(tdata, ydata)
print(pfit)
print(sorted(delta_cost_list))
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
