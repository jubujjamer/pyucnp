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

import matplotlib.pyplot as plt



def gaussian_function(l, mean, std):
    """ Simple gaussian function.
    """
    return 1/(np.sqrt(2*np.pi)*std)*np.exp(-0.5*(l-mean)**2/std**2)


class SourceBeam(object):
    """ Light beam definitions, properties and visualization.
    """
    def __init__(self, beam_shape=None, amplitude=None, wavelength=None,
                 waist=None, width=None):
        """
        Parameters:
        ----------
            beam_shape:  (string) shape of the electric field
                         gaussian:  centered gaussian wave
                         constant:  constant amplitude flat beam
            amplitude:   (float) prefactor of the electric field
            wavelength   (float) [um] wavelength of the electric field
            waist        (float) gaussian beam waist
        """
        __all__ = ['__init__', 'beam_power',  'plot']

        self.beam_shape = beam_shape
        if amplitude is not None:
                self.amplitude = float(amplitude)
        else:
            self.amplitude = None
        if wavelength is not None:
            self.wavelength = float(wavelength)
        else:
            self.wavelength = None
        if waist is not None:
            self.waist = float(waist)
        else:
            self.width = None
        if width is not None:
            self.width = float(width)
        else:
            self.width = None

    def field_power(self, x, y, z):
        """ Calculates the beam power in a given point where point=[x, y, z]
        """
        if self.beam_shape == 'gaussian':
            """ Gaussian beam power.
            """
            zr = np.pi*self.waist**2/self.wavelength
            i0 = self.amplitude**2/(1+(z/zr)**2)
            p0 = np.exp(-2*(x**2+y**2)/(self.waist**2*(1+(z/zr)**2)))
            return i0*p0

        if self.beam_shape == 'hattop':
            """ Constant amplitude beam with the same power as the gaussian.
            """
            radius = np.sqrt(x**2+y**2)
            if self.width is None:
                print("Constant amplitude beam must specify the beam's width.")
                return
            if self.waist is None:
                print("Constant amplitude beam must specify waist for Gaussian Comparison.")
                return
            beam_area = np.pi*(self.width/2)**2
            # print(x, x**2)
            # Why it doesn't work with x**2???
            # if np.abs(x) < self.width/2 and np.abs(y) < self.width/2:
            #     beam_area = np.pi*(self.width/2)**2
            #     P0 = self.amplitude**2*np.pi*self.waist/beam_area
            #     P0 = np.pi*P0/8
            # else:
            #     P0 = 0
            # beam_area = np.pi*(self.width/2)**2
            # P0 = int(radius < self.width/2)*self.amplitude**2*np.pi*self.waist/beam_area
            beam_area = np.pi*(self.width/2)**2
            P0 = self.amplitude**2*np.pi*self.waist/beam_area
            P0 = 0.5*P0/(1+np.exp(10*(radius-(self.width/2))))
            return P0

    def field_integral(self, box):
        """ Integral of the electric field inside an input box.
        Parameters:
        ----------
            box:    (3x2 array) [ [x0, x1], [y0,y1], [z0, z1] ]
        """
        # if self.beam_shape == 'hattop':
        #     def opts0(x, y, z):
        #         if np.sqrt(x**2+y**2) > (self.width/2):
        #             return {'points': [x,y,z]}
        #         else:
        #             return {'points': []}
        # elif self.beam_shape == 'gaussian':
        #     def opts0(x, y, z):
        #         return {'points': []}
        integral = integrate.nquad(self.field_power, box, opts=[{'epsabs': 5e-7}, {}, {}, {} ])
        return integral[0]

    def scalar_field_plot(self, box):
        from mayavi import mlab
        xmin, xmax = box[0]
        ymin, ymax = box[1]
        zmin, zmax = box[2]
        x, y, z = np.ogrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
        field = self.field_power(x, y, z)
        print(field.max())
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
        mlab.pipeline.volume(mlab.pipeline.scalar_field(field), vmin=0)
        mlab.view(azimuth=0,  elevation=90, distance=200, focalpoint=None, roll=0)
        mlab.outline(extent=[0, 100, 0, 400, 0, 100], opacity=1.)
        mlab.show()

class Material(object):
    """ Emissive material and its properties.
    """
    def __init__(self, bands_parameters=None):
        """
        Parameters:
        ----------
            bands_parameters: (list) [[emission bands (nm), c1, c2 ], ...]
        """
        __all__ = ['__init__', 'beam_power',  'plot']
        if bands_parameters is not None:
            center_wavelength = bands_parameters[0]
            self.bands_parameters = bands_parameters


class EmissionMeasurement(object):
    """ Experiment definitions, properties and visualization.
    """
    def __init__(self, source_beam=None, material=None, geometry=None):
        __all__ = ['__init__', 'add',  'plot']
        self.source_beam = source_beam
        self.material = material
        if geometry is not None:
            self.set_geometry(geometry)

    def set_geometry(self, box):
        """ Definition of the geometry of the experiment.
        """
        self.geometry = box
        return box

    def emitted_band_power(self, x, y, z, c1, c2):
        """
        Parameters:
        ----------
        """
        beam_power = self.source_beam.field_power(x, y, z)
        band_power = c1*beam_power + c2*beam_power**2
        return band_power

    def total_emitted_power(self, x, y, z):
        for center, c1, c2 in self.material.bands_parameters:
            return self.emitted_band_power(x, y, z, c1, c2)

    def band_integrated_power(self, c1, c2):
        band_power = integrate.nquad(self.emitted_band_power,
                                     self.geometry, args=(c1, c2))
        return band_power[0]

    def band_spectrum(self, band_parameters, wlens=None):
        """ Band spectrum simulation
        Parameters:
        ----------
            band_parameters: (list) [center, c1, c2]
        """
        center, c1, c2 = band_parameters
        P0 = self.band_integrated_power(c1, c2)
        spectrum = P0*gaussian_function(wlens, center, std=10)
        return np.array(spectrum)

    def total_spectrum(self, wlens):
        """ Total spectrum simulation
        Parameters:
        ----------
            wlens: (list) array of wavelengths
        """
        total_spectrum = np.zeros_like(wlens)
        for band_parameters in self.material.bands_parameters:
            total_spectrum += self.band_spectrum(band_parameters, wlens)
        return total_spectrum



# x, y, z = np.ogrid[-rmax:rmax:100j, -rmax:rmax:100j, -600:600:100j]
# A0 = 1
# w0 = 5
# wlen = 0.5
# zr = np.pi*w0**2/wlen
# i = A0/(1+(z/zr)**2)*np.exp(-(x**2+y**2)/(w0**2 * (1+(z/zr)**2 ) ) )
# mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))
# mlab.pipeline.volume(mlab.pipeline.scalar_field(i), vmin=0, vmax=.5)
# mlab.view(azimuth=0,  elevation=90, distance=800, focalpoint=None, roll=0)
# mlab.outline(extent=[0, 100, 0, 400, 0, 100], opacity=1.)
# mlab.show()
#
#
# def gaussian(l, mean, std):
#     spec = 1/(np.sqrt(np.pi)*std)*np.exp(-(l-mean)**2/std**2)
#     return spec
#
# def calculate_po(Pi=None, c1=0, c2=0):
#     Po = c1*Pi+c2*Pi**2
#     return Po
#
# # l = np.linspace(500, 700, 200)
# # Pi=.5
# # Pa = calculate_po(Pi=Pi, c1=1, c2=1)
# # Pb = calculate_po(Pi=Pi, c1=1, c2=2)
# # spec_a = Pa*2*gaussian(l, 540, 20)
# # spec_b = Pb*gaussian(l, 650, 20)
# # plt.plot(l, spec_a+spec_b)
# # plt.show()
#
#
# def gaussian_beam(x, y, z, A0, w0, wlen):
#     zr = np.pi*w0**2/wlen
#     return A0**2/(1+(z/zr)**2)*np.exp(-(x**2+y**2)/(w0**2 * (1+(z/zr)**2 ) ) )
#
# def output_power(x, y, z, A0, w0, wlen, c1, c2):
#     power_distribution = gaussian_beam(x, y, z, A0, w0, wlen)
#     return c1*power_distribution+c2*power_distribution**2
#
# # total_box_power = integrate.nquad(output_power, [[-rmax,rmax], [-rmax,rmax], [-250, 250]], args=(1, 5, 0.5, 1, 1))
