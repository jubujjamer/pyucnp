 # Check styles directory
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import SpectralData, Spectrum, SpectralDecay

measurement_day = '2019-02-06'
cfg = data.load_cfg(measurement_day, config_file='sample_1.yaml')

spectra_sweep = SpectralData()
spectra_solvents = SpectralData()
spectra_cu = SpectralData()
# idecay_names = cfg_analysis.idecay_curves

## First set up spectral data for later analysis
for spectra in [spectra_sweep, spectra_solvents, spectra_cu]:
    spectra.relevant_bands  = cfg.bands
    spectra.relevant_peaks = cfg.peaks # All visible peaks
    spectra.analysis_peaks = cfg.peaks_reduced # Peaks I want to analyse right now

# ## Load data and store in the spectra class
# # Load the spectral data
# For power sweep
for nsample in cfg.samples_sweep:
    wlens, spectrum = data.load_spectrum(measurement_day, nsample)
    spectrum = Spectrum(wlens, spectrum, counts_to_power=1, excitation_power=1,
                        normalization='background')
    spectra_sweep.addSpectrum(spectrum, index=nsample)
# For samples in different solvents
for nsample in cfg.samples_solvents:
    wlens, spectrum = data.load_spectrum(measurement_day, nsample)
    spectrum = Spectrum(wlens, spectrum, counts_to_power=1, excitation_power=1,
                        normalization='background')
    spectra_solvents.addSpectrum(spectrum, index=nsample)
# For Cu samples
for nsample in cfg.samples_cu:
    wlens, spectrum = data.load_spectrum(measurement_day, nsample)
    spectrum = Spectrum(wlens, spectrum, counts_to_power=1, excitation_power=1,
                        normalization='background')
    spectra_cu.addSpectrum(spectrum, index=nsample)

data.save_pickled(measurement_day, filename='samples_sweep', spectra=spectra_sweep)
data.save_pickled(measurement_day, filename='samples_solvent', spectra=spectra_solvents)
data.save_pickled(measurement_day, filename='samples_cu', spectra=spectra_cu)
