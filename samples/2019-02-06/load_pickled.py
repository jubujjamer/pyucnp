# Check styles directory
import numpy as np
import matplotlib.pyplot as plt

from pyucnp.fitting import robust_best_fit
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import Spectrum


measurement_day = '2019-02-06'
filename = 'samples_cu.sp'
cfg = data.load_data(measurement_day, config_file='sample_1.yaml')
# Open measurements data
spectra = data.load_pickled(measurement_day, filename)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[5, 4], sharex=True)

# Dilution correction
dilcorr = [2.1, 2.12, 2.14, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
dilcorr_iter = iter(dilcorr)
for key, spectrum in spectra.spectra.items():
    corr = next(dilcorr_iter)/dilcorr[0]
    wlens = spectrum.wavelengths
    ydata = spectrum.spectrum_intensity/corr
    axes.plot(wlens, ydata)
plt.show()
