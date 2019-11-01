 # Check styles directory
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay

measurement_day = '2019-02-15'
cfg = data.load_cfg(measurement_day, config_file='sample_1.yaml')

spectra_cu = SpectralData()
# idecay_names = cfg_analysis.idecay_curves

## First set up spectral data for later analysis
spectra_cu.relevant_bands  = cfg.bands
spectra_cu.relevant_peaks = cfg.peaks # All visible peaks
spectra_cu.analysis_peaks = cfg.peaks_reduced # Peaks I want to analyse right now

# ## Load data and store in the spectra class
# # Load the spectral data
# For Cu samples
for nsample in cfg.spectra:
    wlens, spectrum = data.load_spectrum(measurement_day, nsample)
    spectrum = Spectrum(wlens, spectrum, counts_to_power=1, excitation_power=1,
                        normalization='background')
    spectra_cu.addSpectrum(spectrum, index=nsample)

data.save_pickled(measurement_day, filename='samples_cu', spectra=spectra_cu)
