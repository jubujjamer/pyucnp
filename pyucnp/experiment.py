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
            self.waist = None
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
            radius = x**2+y**2
            if self.width is None:
                print("Constant amplitude beam must specify the beam's width.")
                return
            if self.waist is None:
                print("Constant amplitude beam must specify waist for Gaussian Comparison.")
                return
            # Why it doesn't work with x**2???
            # beam_area = np.pi*(self.width/2)**2
            # P0 = (radius < self.width/2)*self.amplitude**2*np.pi*self.waist/beam_area
            ######## A cylinder (integral does not converge)
            # beam_area = np.pi*(self.width/2)**2
            # P0 = np.ones_like(z)*self.amplitude**2*np.pi*self.waist**2/beam_area
            # cut_factor = 1/(1+np.exp(20*(radius-(self.width/2)**2)))
            # P0 = 0.5*P0*cut_factor
            # Rectangular beam (easier integration)
            beam_area = (self.width)**2
            s = self.width/2
            if np.abs(x) < s and np.abs(y) < s:
                P0 = 0.5*np.ones_like(z)*self.amplitude**2*np.pi*self.waist**2/beam_area
            else:
                P0 = 0
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
        integral = integrate.nquad(self.field_power, box, opts=[{}, {}, {}, {} ])
        return integral[0]

    def scalar_field_plot(self, box):
        from mayavi import mlab
        xmin, xmax = box[0]
        ymin, ymax = box[1]
        zmin, zmax = box[2]
        x, y, z = np.ogrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
        field = self.field_power(x, y, z)
        mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
        mlab.pipeline.volume(mlab.pipeline.scalar_field(field), vmin=0, vmax=.005)
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


class SpectralData(object):
    """ Class to manage emmission, power and decay curves.
    """
    def __init__(self, relevant_peaks=None, analysis_peaks=None,
                relevant_bands=None):
        """
        Parameters:
        ----------
        relevant_peaks      list
                            list containing the most prominent wavelenghts.
        analysis_peaks      list
                            Wavelenght in wich to focus in this spectrum.
        relevant_bands      dict
                            Band definition {'Band name': [start_wl, end_wl], }

        """
        __all__ = ['__init__']
        self.relevant_bands = relevant_bands
        self.relevant_peaks = relevant_peaks
        self.analysis_peaks = analysis_peaks
        self.spectral_decays = dict() # keys of this dict are measurement indexes and wavelength
        self.spectra = dict() # keys of this dict are measurement indexes

    def addSpectrum(self, spectrum, index):
        """"Adds a spectrum to spectra container. Index should be consistent
        with the keys of cfg.spectrum_data for each measurement."""
        self.spectra[index] = spectrum

    def addSpectralDecay(self, spectral_decay, index, wavelength):
        """"Adds a time decay to spectra container. Index should be consistent
        with the keys of cfg.spectrum_data for each measurement."""
        if index not in self.spectral_decays.keys():
            self.spectral_decays[index] = dict()
        self.spectral_decays[index][wavelength] = spectral_decay

    def intensityPeaks(self, index):
        """ Calculates the power in isolated spectral peaks.

        It takes the peaks from 'relevant_peaks' and outputs the spectrum
        intensity for each of them by itegrating over a small band.

        Parameters
        ----------
        index : type
            Description of parameter `index`.

        Returns
        -------
        type
            Description of returned object.

        """
        try:
            spectrum = self.spectra[index]
        except:
            raise Exception('Spectrum with index %i was not added' % index)
        peaks_intensities = [spectrum.integrate_band(peak-2, peak+2) for peak in self.relevant_peaks]
        return np.array(self.relevant_peaks), np.array(peaks_intensities)

    def intensityBands(self, index):
        try:
            spectrum = self.spectra[index]
        except:
            raise Exception('Spectrum with index %i was not added' % index)
        names, limits = zip(*self.relevant_bands.items())
        bands_intensities = [spectrum.integrate_band(l, h) for l, h in limits]
        bands_dict = dict(zip(names, np.array(bands_intensities)))
        return bands_dict

    def limitingSlopes(self, wavelength):
        """ Calculates limiting slopes in a log log plot.
        """
        from pyucnp.fitting import fit_line
        ## Defines wich points correspond to low and high power linear limits.
        hpslice = slice(-5, -1)
        lpslice = slice(12, 20)
        indexes, spectra = zip(*self.spectra.items())
        excitations = [s.excitation_power for s in spectra]
        intensity_curve = [s.peak_intensity(wavelength) for s in spectra]
        xlog, ylog = [np.log10(excitations), np.log10(intensity_curve)]
        lp_params = fit_line(xlog[lpslice], ylog[lpslice])
        hp_params = fit_line(xlog[hpslice], ylog[hpslice])
        return lp_params['m'].value, hp_params['m'].value


    def decay_parameters(self, index, wavelength):
        """ Calculates exponential decay parameters for a given wavelength.
        """

        from pyucnp.fitting import robust_best_fit
        try:
            spectral_decay = self.spectral_decays[index][wavelength]
        except:
            raise Exception('Spectral decay with index %i was not added' % index)
        if spectral_decay.a1 is not None:
            return spectral_decay.a1, spectral_decay.tau1, spectral_decay.tau1, spectral_decay.tau2
        else:
            tdata = spectral_decay.time
            idata = spectral_decay.idata
            result = robust_best_fit(tdata, idata, model='double_neg')
            a1 = result.params['a1'].value
            tau1 = 1000/result.params['ka'].value
            try:
                a2 = result.params['a2'].value
                tau2 = 1000/result.params['kUC'].value
            except:
                a2 = 0
                tau2 = 1
            spectral_decay.a1 = a1
            spectral_decay.tau1 = tau1
            spectral_decay.a2 = a2
            spectral_decay.tau2 = tau2
            return a1, tau1, a2, tau2


    def promediate_decays(self, index, wlen_start, wlen_end, weighted=False):
        in_band = [w>=wlen_start and w<wlen_end for w in self.relevant_peaks]
        peaks = np.array(self.relevant_peaks)
        if weighted:
            wlens, weights = self.intensityPeaks(index=33)
            weights /= np.sum(weights)
        else:
            weights = np.ones(len(in_band))
        for weight, wlen in zip(weights, peaks[in_band]):
            decay = self.spectral_decays[index][wlen]
            idata = decay.idata
            if 'mean_decay' not in locals():
                time = decay.time
                mean_decay = np.zeros_like(time)
            mean_decay += idata
        return time, mean_decay

class SpectralDecay(object):
    """ Class to manage emission, power and decay curves.
    """
    def __init__(self, time=None, idata=None, excitation_power=None):
        """ SpectralDecay initializer.
        Parameters
        ----------
        time                array
                            Time data in a 1D array.
        idata               array
                            Intensity axis for the spectra decay.
        excitation_power    float
                            Excitation power for the given measurement.

        Returns
        -------
        type
            An instance of ScaterDecay.
        """
        self.idata = idata
        self.time = time
        self.excitation_power = excitation_power
        self.a1 = None
        self.tau1 = None
        self.a2 = None
        self.tau2 = None

    def isEmpty(self):
        if self.time is None:
            return False
        return True


class Spectrum(object):
    """ Sectrum class to hold all intensity measurements.
        Parameters
        ----------
        wavelenghts : list or array
            Wavelenght axis of the spectrum.
        spectrum_intensity : type
            Intensity axis of the spectrum, should have the same number of points than wavlengths.
        counts_to_power : float
            Conversion factor from counts to power in mW to conver the intensity measurements.
        excitation_power : float
            Power of the excitation source.
        normalization : string
            The type of normalization desired. It can be:
            'background': background correction.
            'none': for no correction or scaling

        Returns
        -------
        Spectrum
            Instance of the Spectrum class.

        """
    def __init__(self, wavelenghts, spectrum_intensity, counts_to_power=None, excitation_power=None, normalization='none'):

        self.excitation_power = excitation_power
        self.counts_to_power = counts_to_power
        self.normalization = normalization

        if len(wavelenghts) != len(spectrum_intensity):
            raise Exception('Number of points in wavelenghts should equal number of points in intensities.')
        else:
            self.wavelengths = np.array(wavelenghts)
            self.spectrum_intensity = np.array(spectrum_intensity)
            try:
                float(self.counts_to_power)
                self.spectrum_intensity *= self.counts_to_power
            except:
                raise Exception('Counts to power ratio is not a valid number.')

    def integrate_band(self, start_wl, end_wl):
        from scipy.integrate import simps
        start_pind = int(np.where(self.wavelengths == start_wl)[0])
        end_pind = int(np.where(self.wavelengths == end_wl)[0])
        integral_power = simps(self.spectrum_intensity[start_pind:end_pind])
        return integral_power

    def get_values(self):
        """ Returns the spectrum values considering the transformation chosen.

        Returns
        -------
        wavelenghts: array
            Wavelenght axis of the spectrum.
        spectrum_corrected: array
            Intensity axis corrected for the given spectrum.

        """
        print('Showing spectrum with %s transformation.' % self.normalization)
        if self.normalization == 'background':
            background = np.mean(self.spectrum_intensity[0:10])
            spectrum_corrected = self.spectrum_intensity - background + 1
            return self.wavelenghts, spectrum_corrected

        elif self.normalization == 'none':
            return self.wavelenghts, spectrum_corrected

    def peak_intensity(self, wavelength):
        """ Intensity of a given peak.

        Returns the intensity of a desired wavelength. It integrates a small band (6 nm by default)
        around the center wavelength sacled acording to the conversion factor counts_to_power.

        Parameters
        ----------
        wavelength : array
            Wavelenght axis of the spectrum..

        Returns
        -------
        peak_amp: float
            Amplitude of the given peak.
        """
        peak_amp = self.integrate_band(start_wl=wavelength-3, end_wl=wavelength+3)
        return peak_amp
