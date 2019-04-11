 # Check styles directory
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay
import logging

logging.basicConfig(level=logging.WARNING)

daystr = '2019-02-15'
spectra = SpectralData.from_folder(daystr='2019-02-15', sample='sample1')
print(spectra.power_peaks(10))
