 # Check styles directory
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay
import logging

logging.basicConfig(level=logging.DEBUG)

measurement_day = '2019-02-15'
spectra = SpectralData.from_folder(measurement_day)
spectra[10]
