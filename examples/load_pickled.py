# Check styles directory
import numpy as np
import matplotlib.pyplot as plt

from pyucnp.fitting import robust_best_fit
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import Spectrum


measurement_day = '2019-02-06'
filename = 'sample_1.sp'
cfg = data.load_cfg(measurement_day, )
# Open measurements data
spectra = data.load_pickled(measurement_day, filename)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[5, 4], sharex=True)

spectrum = spectra.spectra[2]
wlens = spectrum.wavelengths
ydata = spectrum.spectrum_intensity
axes.plot(wlens, ydata)
plt.show()
