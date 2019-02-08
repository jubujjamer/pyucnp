 # Check styles directory
import numpy as np
import pickle

from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay

measurement_day = '2019-02-06'
cfg = data.load_data(measurement_day, )

spectra = SpectralData()
# idecay_names = cfg_analysis.idecay_curves

## First set up spectral data for later analysis
spectra.relevant_bands  = cfg.bands
spectra.relevant_peaks = cfg.peaks # All visible peaks
spectra.analysis_peaks = cfg.peaks_reduced # Peaks I want to analyse right now

# ## Load data and store in the spectra class
# # Load the spectral data
for nsample in cfg.samples_sweep:
    spectrum = Spectrum(*data.load_spectrum(measurement_day, nsample),
                        counts_to_power=1,
                        excitation_power=1,
                        normalization='background')
    spectra.addSpectrum(spectrum, index=nsample)

# # Then load the time series
# for wlen, name in zip(cfg_analysis.peaks, cfg_analysis.idecay_curves):
#     tdata, idata = dt.load_idecay(daystr, name, nbins=80, ndata=-1, TS=6.4E-8)
#     excitation_power = float(cfg.spectrum_data[33][3])
#     spectral_decay = SpectralDecay(time=tdata, idata=idata, excitation_power=excitation_power)
#     spectra.addSpectralDecay(spectral_decay, index=33, wavelength=wlen)

# # Force fittings to save parameters and do all only one time
# for wlen in spectra.relevant_peaks:
#     parameters = spectra.decay_parameters(index=33, wavelength=wlen)

# with open('filename', 'wb') as outfile:
#     pickle.dump(spectra, outfile)
