#!/usr/bin/python
# -*- coding: utf-8 -*-
""" geometry.py
Last update: 01/04/2017

Geometrical analysis of beams and coloidal or dryed UNCP measurement.
Usage:

"""

import numpy as np
import matplotlib.pylab as plt
# from mayavi import mlab
from scipy import integrate

from pyucnp.experiment import *


field_amplitude = 2

fix, axes = plt.subplots(1, 3)
axes_iter = iter(axes)
rmax_iter = iter([2, 4, 10])
zmax = 1.

for waist in [.1, .5, 1.]:
    constant_beam = SourceBeam(beam_shape='hattop', amplitude=field_amplitude, wavelength=0.5, width=1., waist=waist)
    gaussian_beam = SourceBeam(beam_shape='gaussian', amplitude=field_amplitude, wavelength=0.5, waist=waist)

    rmax = rmax_iter.next()
    box = np.array([[-rmax, rmax], [-rmax, rmax], [-zmax, zmax]])
    power_flux = constant_beam.field_integral(box)
    print("Gaussian power flux", gaussian_beam.field_integral(box))
    print("Constant power flux", power_flux)

    material = Material([[540, 4, 4], [650, 1, 4]])
    geometry = box
    experiment_gauss = EmissionMeasurement(gaussian_beam, material, geometry)
    experiment_constant = EmissionMeasurement(constant_beam, material, geometry)

    wlens = np.linspace(500, 700, 200)
    band_parameters = [540, 1, 2]
    spectrum_gauss = experiment_gauss.total_spectrum(wlens=wlens)
    spectrum_constant = experiment_constant.total_spectrum(wlens=wlens)
    print('plot')
    ax = axes_iter.next()
    ax.plot(wlens, spectrum_gauss/spectrum_gauss.max(), 'k--',linewidth=1.5)
    ax.plot(wlens, spectrum_constant/spectrum_constant.max(), 'g-.',linewidth=2.5)
    ax.legend(['Gaussian waist: %.2f' % waist, 'Constant'])
    ax.grid()
    ax.set_xlabel('Longitud de onda (nm)')
    ax.set_ylabel('Intensidad (U.A.)')
    ax.set_title('Power flux %.2f' % power_flux)
plt.show()
