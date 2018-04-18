# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 14:34:04 2016
@author: Juan
"""
import numpy as np
import lmfit
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import csv
import itertools as it

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.cm as cm
import colour.plotting as cplt
import colour


from pyucnp.fitting import fit_line, fit_power


def wlen_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


def setup_matplotlib():
    """ Setups matplotlib font and sizes.
    """
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rc('legend',**{'fontsize':14})
    plt.rcParams['legend.numpoints'] = 1

def annotate(ax, text, xytext):
    """ Makes an annotation in the given position.
    """
    ax.annotate(text, xy=(-20, 45),  xycoords='data', xytext=xytext,
            textcoords='axes fraction', horizontalalignment='right', verticalalignment='top')


def plot_stationary(power_sweep_list, power_labels, wavelength_list, filename=None):
# def plot_stationary(spectrum_list, wavelength_list, filename=None):
    setup_matplotlib()
    fig, axes = plt.subplots(len(wavelength_list), 1, figsize=[8, 6], sharey=True)
    axiter = iter(fig.axes)
    for wlen in wavelength_list:
        ax = next(axiter)
        hpslice = slice(-5, -1)
        lpslice = slice(7, 20)
        pamps = np.array(power_sweep_list[wlen])
        x, y = [10*np.log10(power_labels), 10*np.log10(pamps)]
        lp_params = fit_line(x[lpslice], y[lpslice])
        hp_params = fit_line(x[hpslice], y[hpslice])
        ax.plot(x, y, label='%3i nm' % wlen, marker='o', color=wlen_to_rgb(wlen),
                markersize=4.5, linewidth = .0)
        ax.plot(x, lp_params['b']+lp_params['m']*x, 'k--', linewidth=1.2)
        ax.plot(x, hp_params['b']+hp_params['m']*x, 'k--', linewidth=1.2)
        ax.set_title('$\\lambda = $%s nm' % wlen)
        annotate(ax, '$\\alpha_1$ = %.2f' % lp_params['m'], xytext=(.8, .15))
        annotate(ax, '$\\alpha_2$ = %.2f' % hp_params['m'], xytext=(.8, .1))
        # ax.legend(bbox_to_anchor=(0.35, 0.15), loc=2, borderaxespad=0.)
        ax.set_xlim(-30, -6)
        ax.set_ylim(40, 82)
        ax.set_xlabel('Excitation power (dBm)')
        ax.set_ylabel('Emission power (log(), U.A.)')
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_3d(spectrum_list, wavelength_list, filename=None):
    setup_matplotlib()

    fig = plt.figure(figsize=[12,12])
    ax = fig.gca(projection='3d')
    verts = []
    zs = [s['label'] for s in spectrum_list]
    colors = [cm.hot(p/5000.) for p in zs]
    for spectrum_dict in spectrum_list:
    #     ys = np.random.rand(len(x))
        x = spectrum_dict['x']
        y = spectrum_dict['y']
        # x = spec_dict[z][0]
        # y = spec_dict[z][1]
        y[0], y[-1] = 0, 0
        verts.append(list(zip(x, y)))

    poly = PolyCollection(verts,  facecolors=colors)
    poly.set_alpha(1.)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    ax.set_xlabel('Long. de onda (nm)')
    ax.set_xlim3d(350, 700)
    ax.set_ylabel('Potencia incidente (U.A.)')
    ax.set_ylim3d(0, 5000)
    ax.set_zlabel('Potencia de salida (U.A.)')
    ax.set_zlim3d(0, 5.5E7)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def cie(spectrum_list):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    verts = list()
    codes = list()
    # codes.append(Path.MOVETO)
    fig =  cplt.chromaticity_diagram_plot_CIE1931(standalone=False)
    for i, spectrum in enumerate(spectrum_list):
        spectrum['label']
        spd_data = dict(zip(spectrum['x'], spectrum['y']/max(spectrum['y'])))
        # sample_spd_data = {522: 0.048,523: 0.152, 524: 0.053, 525: 0.054, 600: 0}
        spd = colour.SpectralPowerDistribution(spd_data)
        cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        # # illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['D65']
        # # Calculating the sample spectral power distribution *CIE XYZ* tristimulus values.
        XYZ = colour.spectral_to_XYZ(spd, cmfs)
        xy =  colour.XYZ_to_xy(XYZ)
        # Plotting the *xy* chromaticity coordinates.
        x, y = xy
        # plt.plot(x, y, 'o-', color='black')
        verts.append((x, y))
        if i == 0:
            codes.append(Path.MOVETO)
        else: codes.append(Path.CURVE3)
    print(len(codes), len(verts))
    # codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='none', lw=1.5)
    ax.add_patch(patch)
    # # Annotating the plot.
    # plt.annotate('',
    #                xy=xy,
    #                xytext=(-50, 30),
    #                textcoords='offset points',
    #                arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-0.2'))

    # Displaying the plot.
    cplt.render(
        standalone=True,
        limits=(-0.1, 0.9, -0.1, 0.9),
        x_tighten=True,
        y_tighten=True)
