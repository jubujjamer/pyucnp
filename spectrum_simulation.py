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
waist = .1
constant_beam = SourceBeam(beam_shape='hattop', amplitude=field_amplitude, wavelength=0.5, width=1, waist=waist)
gaussian_beam = SourceBeam(beam_shape='gaussian', amplitude=field_amplitude, wavelength=0.5, waist=waist)

# print("Gauss power flux", gaussian_beam.field_integral(np.array([[-50, 50], [-50, 50], [-1, 1]])))
rmax = 1
zmax = 1
box = np.array([[-rmax, rmax], [-rmax, rmax], [-zmax, zmax]])
# print("Gaussian power flux", gaussian_beam.field_integral(box))
# print("Constant power flux", constant_beam.field_integral(box))
gaussian_beam.scalar_field_plot(box)

# material = Material([[540, 4, 4], [650, 1, 4]])
# rmax = 5
# # geometry = np.array([[-rmax, rmax], [-rmax, rmax], [-10, 10]])
# geometry = box
# experiment_gauss = EmissionMeasurement(gaussian_beam, material, geometry)
# experiment_constant = EmissionMeasurement(constant_beam, material, geometry)
#
# wlens = np.linspace(500, 700, 200)
# band_parameters = [540, 1, 2]
# spectrum_gauss = experiment_gauss.total_spectrum(wlens=wlens)
# spectrum_constant = experiment_constant.total_spectrum(wlens=wlens)
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(wlens, spectrum_gauss/spectrum_gauss.max(), 'k--',linewidth=1.5)
# ax.plot(wlens, spectrum_constant/spectrum_constant.max(), 'g-.',linewidth=2.5)
# ax.legend(['Gaussian waist: %.2f' % waist, 'Constant'])
# ax.grid()
# ax.set_xlabel('Longitud de onda (nm)')
# ax.set_ylabel('Intensidad (U.A.)')
# plt.show()
