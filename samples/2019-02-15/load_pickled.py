# Check styles directory
import numpy as np
import matplotlib.pyplot as plt

from pyucnp.fitting import robust_best_fit
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import Spectrum


measurement_day = '2019-02-15'
filename = 'samples_cu.sp'
cfg = data.load_cfg(measurement_day, config_file='sample_1.yaml')
# Open measurements data
spectra = data.load_pickled(measurement_day, filename)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[5, 4])
# Dilution correction
spectrum = spectra.spectra[1]

def correct_shift(spec_ref, spec):
    corr = np.correlate(spec_ref, spec, mode='full')/np.sum(spec_ref.ravel()**2)
    max_index = np.where(corr == np.max(corr))[0]
    shift = 1+max_index-len(spec_ref)
    return np.roll(spec, shift)

wlens = spectrum.wavelengths
ydata_ref = spectrum.spectrum_intensity
init_volume = cfg.volumes[1]
gy_ratio = list()
concentrations = list()

for key in cfg.spectra_reduced:
    spectrum = spectra.spectra[key]
    dilution_factor = cfg.volumes[key]/init_volume
    crbh, ccu = cfg.CRBHCu[key]
    min_concentration = np.minimum(crbh, ccu)
    wlens = spectrum.wavelengths
    start_index = np.where(wlens==500)[0][0]
    wlens = wlens[start_index:]
    # norm_index = np.where(wlens==659)[0][0]
    ydata = spectrum.spectrum_intensity[start_index:]
    ydata = correct_shift(ydata_ref, ydata)*dilution_factor
    concentrations.append(min_concentration)
    gy_ratio.append(spectrum.integrate_band(570, 630)/spectrum.integrate_band(545, 552))
    label = '%i RBH %.1f, Cu %.1f (uM)' % (key, 1E6*crbh, 1E6*ccu)
    axes[0].plot(wlens, ydata, label=label)
axes[1].set_title('Concentración limitante vs. cociente RBH/G')
axes[1].set_xlabel('Concentracion limitante (M)')
axes[1]. set_ylabel('Relación banda RBH / Verde')
axes[1].plot(concentrations, gy_ratio, 'ro')
axes[0].set_xlabel('Long. de onda (nm)')
axes[0]. set_ylabel('Intensidad')
axes[0].legend(ncol=5)
plt.show()
