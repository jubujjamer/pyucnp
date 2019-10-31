 # Check styles directory
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay
import logging

logging.basicConfig(level=logging.WARNING)

measurement_day = '2019-02-15'
spectra = SpectralData.from_folder(measurement_day)

spectra[10]
# sp = spectra.get_spectrum_from_power(10)
#
# din = sp.get_decay_at(532)
