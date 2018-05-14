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
# import matplotlib.colors
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import PolyCollection, PatchCollection
# import matplotlib.patches as patches
# import matplotlib.cm as cm
# import colour.plotting as cplt
# import colour


from pyucnp.fitting import fit_line, fit_power, robust_fit


def wlen_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an approximate RGB color
    value. The wavelength must be given in nanometers in the range from 380 nm
    through 750 nm (789 THz through 400 THz).

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


def normalize(tdecay, idecay, iss=None, mode='iss'):
    """ Normalize a given lifetime curve area by the spectrum peak. Given the
    area A under the lifetime curve I(t) and its stationary spectrum peak, it's
    new value will be I(t)*Iss/A..

    Parameters
    ----------
    tdecay : array
        Time series for the idecay curve.
    idecay : array
        Decay curve for a given wavelenght and excitation power.
    iss : float
        Stationary powe peak for the given wavelength and excitation power.

    Returns
    -------
    array, array
        Time series and normalized decay curve.

    """
    if mode == 'iss':
        from scipy.integrate import simps
        A = simps(idecay, tdecay)
        inorm = idecay*iss/A

    elif mode == 'maximum':
        inorm = idecay/np.max(idecay)

    elif mode == 'sum':
        # from scipy.integrate import simps
        inorm = idecay/sum(idecay)

    return tdecay, inorm

def annotate(ax, text, xytext):
    """ Anonotates a given axis at a given position.

    Parameters
    ----------
    ax : (matplotlib axes)
        Description of parameter `ax`.
    text : string
        Description of parameter `text`.
    xytext : tuple
        Description of parameter `xytext`.

    Returns
    -------
    None

    """
    ax.annotate(text, xy=(-20, 45),  xycoords='data', xytext=xytext,
            textcoords='axes fraction', horizontalalignment='right', verticalalignment='top')


def plot_idecays(tdecay, idecay_dict, wlen_list=None, plot_fit=False, ax=None):
    """Short summary.

    Parameters
    ----------
    tdecay : type
        Description of parameter `tdecay`.
    idecay_dict : type
        Description of parameter `idecay_dict`.
    wlen_list : type
        Set this if you don't want to plot all wavelengths.
    plot_fit : type
        Description of parameter `plot_fit`.
    ax : type
        Description of parameter `ax`.

    Returns
    -------
    type
        Description of returned object.

    """
    if ax is None:
        ax = plt.gca()
    label_iter= iter(['%s nm' % wlen for wlen in idecay_dict.keys()])
    for key, idecay  in idecay_dict.items():
        if key not in wlen_list:
            continue
        label = next(label_iter)
        color = wlen_to_rgb(key)
        ax.plot(1E3*tdecay, idecay, color=color,
                marker='o', markersize=3., linestyle='None', label=label)
        if plot_fit:
            result = robust_fit(tdecay, idecay, model='double_neg')
            ax.plot(1E3*tdecay, result.best_fit, color='k',
                    linestyle='--', linewidth=0.8)


def plot_stationary(power_list, power_labels, wlen_list, filename=None):
    """ plot loglog stationary powers.

    Parameters
    ----------
    power_sweep_list : type
        Description of parameter `power_sweep_list`.
    power_labels : type
        Description of parameter `power_labels`.
    wavelength_list : type
        Description of parameter `wavelength_list`.
    filename : type
        Description of parameter `filename`.

    Returns
    -------
    type
        Description of returned object.

    """
    fig, axes = plt.subplots(len(wlen_list), 1, figsize=[8, 6], sharey=False)
    axiter = iter(fig.axes)
    for wlen in wlen_list:
        ax = next(axiter)
        hpslice = slice(-5, -1)
        lpslice = slice(12, 20)
        pamps = np.array(power_list[wlen])
        x, y = [10*np.log10(power_labels), 10*np.log10(pamps)]
        lp_params = fit_line(x[lpslice], y[lpslice])
        hp_params = fit_line(x[hpslice], y[hpslice])
        ax.plot(x, y, label='%3i nm' % wlen, marker='o', color=wlen_to_rgb(wlen),
                markersize=4.5, linewidth = .0)
        ax.plot(x, lp_params['b']+lp_params['m']*x, 'k--', linewidth=1.2)
        ax.plot(x, hp_params['b']+hp_params['m']*x, 'k--', linewidth=1.2)
        # ax.set_title('$\\lambda = $%s nm' % wlen)
        annotate(ax, '$\\alpha_1$ = %.2f' % lp_params['m'], xytext=(.8, .35))
        annotate(ax, '$\\alpha_2$ = %.2f' % hp_params['m'], xytext=(.8, .2))
        # ax.legend(bbox_to_anchor=(0.35, 0.15), loc=2, borderaxespad=0.)
        ax.set_xlim(-30, -6)
        print(np.max(y))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_xlabel('Excitation power (dBm)')
        ax.set_ylabel('Emission power (log(), U.A.)')
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_3d(spectrum_list, wavelength_list, filename=None):
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

def plot_spectrums(spectrum_list, filename=None):
    fig, (ax) = plt.subplots(1, 1, figsize=[8, 4])
    colors = iter([cm.hot(p*0.1) for p in range(len(spectrum_list))])
    for spectrum_dict in spectrum_list:
        x = spectrum_dict['x']
        y = spectrum_dict['y']
        y /= np.max(y)
        ax.plot(x, y, color=next(colors))
    ax.set_xlabel('Long. de onda (nm)')
    ax.set_ylabel('Potencia incidente (U.A.)')
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_times(wavelengths, datasets_list, cfg):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 10])
    names = list(cfg.datasets.values())[0]
    peaks_iter = iter([n.split('_')[1] for n in names])

    delta = 0.0
    label_iter= iter(['379 nm', '490 nm', '546 nm', '654 nm'])
    for tdata, ydata, result in datasets_list[0]:
        delta += .2
        label = next(label_iter)
        peak = next(peaks_iter)
        ax2.plot(1E3*tdata, ydata+delta, color=wlen_to_rgb(peak), marker='o', markersize=1.2)
        ax2.plot(1E3*tdata, result.best_fit+delta, color=wlen_to_rgb(peak), marker='o', markersize=1.2, label=label)
        ax2.set_xlim([0, .35])

    ax1.set_ylabel("Amplitud (U.A.)")
    ax2.set_ylabel("Amplitud (U.A.)")
    ax1.set_xlabel("Longitud de onda (nm)")
    ax2.set_xlabel("Tiempo (ms)")
    ax2.legend(ncol=4, loc='upper right')

    plt.tight_layout(pad=0., w_pad=0.5, h_pad=0.5)
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
    # codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='none', lw=1.5)
    ax.add_patch(patch)
    cplt.render(
        standalone=True,
        limits=(-0.1, 0.9, -0.1, 0.9),
        x_tighten=True,
        y_tighten=True)

def normalized_cie(ax, spectrum_dict):
    # fig =  cplt.chromaticity_diagram_plot_CIE1931(standalone=False)
    spd_data = spectrum_dict
    # sample_spd_data = {522: 0.048,523: 0.152, 524: 0.053, 525: 0.054, 600: 0}
    spd = colour.SpectralPowerDistribution(spd_data)
    cmfs = colour.STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    # # illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['D65']
    # # Calculating the sample spectral power distribution *CIE XYZ* tristimulus values.
    XYZ = colour.spectral_to_XYZ(spd, cmfs)
    # Plotting the *xy* chromaticity coordinates.
    x, y =  colour.XYZ_to_xy(XYZ)
    ax.plot(x, y, 'o-', color='black')


def delayed_cie(idecay_dict):
    """ Plots .

    Parameters
    ----------
    idecay_dict : dict
        Dictionaty of decay curves {wlen: icurve}.

    Returns
    -------
    type
        plots cie spectrum for each delayed time.

    """
    iss_delayed = dict() # dict of intensities {wavelenght: intensity}
    wlen_range = range(350, 700)
    delay_range = range(0, 60, 5)
    fig =  cplt.chromaticity_diagram_plot_CIE1931(standalone=False)
    ax = fig.add_subplot(111)

    for delay in delay_range:
        for wlen in wlen_range:
            if wlen in idecay_dict.keys():
                idecay = idecay_dict[wlen]
                iss_delayed[wlen] = idecay[delay]
            else:
                iss_delayed[wlen] = 0
        normalized_cie(ax, iss_delayed)
    cplt.render(standalone=True, limits=(-0.1, 0.9, -0.1, 0.9), x_tighten=True,
                    y_tighten=True)
    plt.show()

def plot_3d(spectrum_list, wavelength_list, filename=None):

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

def idecays3d(tdecay, idecay_dict, filename=None):

    fig = plt.figure(figsize=[12,12])
    ax = fig.gca(projection='3d')
    zs = iter([float(w) for w in idecay_dict.keys()])
    verts = list()
    for key, idecay in idecay_dict.items():
        wlen = next(zs)
        x = 1E3*tdecay[:-50]
        y = np.log10(idecay[:-50])
        ax.plot(x,np.ones_like(x)*wlen, y, color=wlen_to_rgb(wlen))

    ax.set_xlabel('Time (ms)')
    # ax.set_xlim3d(0, 1.2)WSS
    ax.set_ylabel('Wavelength (nm))')
    ax.set_ylim3d(350, 700)
    ax.set_zlabel('Intensity (A.U.)')
    # ax.set_zlim3d(0, 5.5E3)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_fits(fit_dict):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[10, 7], sharex=True)
    f_list = list()
    kUC_list = list()
    ka_list = list()
    wlen_list = [float(w) for w in fit_dict.keys()]
    for key, result in fit_dict.items():
        a_ka = result.params['a1'].value
        a_kuc = result.params['a2'].value
        f_list.append(-a_ka/(-a_ka+a_kuc))
        kUC_list.append(result.params['kUC'].value/1000)
        ka_list.append(result.params['ka'].value/1000)
    wlens = np.array(wlen_list)

    bar_width = 2.5
    ax1.bar(wlens+bar_width, f_list, bar_width, color='C0', edgecolor='k')
    ax1.set_ylabel('f')
    ax1.set_ylim([0.7, 1.05])
    ax2.bar(wlens+bar_width, ka_list, bar_width, color = 'C1', edgecolor='k')
    ax2.set_ylabel('ka (1/ms)')
    ax3.bar(wlens+bar_width, kUC_list, bar_width, color = 'C2', edgecolor='k')
    ax3.set_ylabel('kUC (1/ms)')
    plt.subplots_adjust(hspace=0.)
    plt.show()


def plot_fits_v2(fit_dict, var='f', ax=None):
    if ax is None:
        ax = plt.gca()
    f_list = list()
    kUC_list = list()
    ka_list = list()
    wlen_list = [float(w) for w in fit_dict.keys()]
    for key, result in fit_dict.items():
        a_ka = result.params['a1'].value
        a_kuc = result.params['a2'].value
        f = -a_ka/(-a_ka+a_kuc)
        f_list.append(-a_ka/(-a_ka+a_kuc))
        if f < 0.99:
            kUC_list.append(result.params['kUC'].value/1000)
        else:
            kUC_list.append(0)
        ka_list.append(result.params['ka'].value/1000)
    wlens = np.array(wlen_list)
    bar_width = 2.5
    if var == 'f':
        bp = ax.bar(wlens+bar_width, f_list, bar_width,
                    facecolor='C0', edgecolor='k')
    elif var == 'ka':
        bp = ax.bar(wlens+bar_width, ka_list, bar_width,
                    facecolor = 'C1', edgecolor='k')
    elif var == 'kUC':
        bp = ax.bar(wlens+bar_width, kUC_list,
                    bar_width, color = 'C2', edgecolor='k')
    # ax3.set_ylabel('kUC (1/ms)')
    # ax3.set_xlabel('Wavelength (nm)')
    # plt.subplots_adjust(hspace=0.1)
    # plt.show()
    return bp

def plot_mean_taus(tdecay, mtime_dict, ax=None):
    if ax is None:
        ax = plt.gca()
    taus = np.array(list(mtime_dict.values()))
    wlens = np.array(list(mtime_dict.keys()))
    xvalues = np.arange(len(wlens))*5
    bar_width = 3
    barlist = ax.bar(xvalues+ bar_width, 1E3*taus, bar_width, edgecolor='k')
    wlens_iter = iter(wlens)
    for i in range(len(barlist)):
        wlen = next(wlens_iter)
        barlist[i].set_facecolor(wlen_to_rgb(wlen))
    return barlist

def plot_spectrum(lamdas, spectrum, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(lamdas, spectrum)
