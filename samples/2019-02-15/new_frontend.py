 # Check styles directory
import matplotlib.pyplot as plt
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay
import logging

logging.basicConfig(level=logging.WARNING)

measurement_day = '2019-02-11'
spectra = SpectralData.from_folder(measurement_day)

spectra.details()
# sp = spectra[2]
# # sp = spectra.get_spectrum_from_power(10)
# plt.plot(sp.get_decay_at(410).idata)
# plt.show()
