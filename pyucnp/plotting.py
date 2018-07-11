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
from matplotlib.ticker import FuncFormatter, MaxNLocator

# import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection, PatchCollection
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

def normalize_spectrum(lambdas, spectrum):

    background = np.mean(spectrum[0:10])
    spectrum_corr = spectrum - background + 1
    return lambdas, spectrum_corr


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


def plot_idecays(tdecay, idecay_dict, wlen_list=None, plot_fit=False,
                ax=None, mode=None):
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
    label_iter= iter(['%s nm' % wlen for wlen in wlen_list])
    for key, idecay  in idecay_dict.items():
        if key not in wlen_list:
            continue
        label = next(label_iter)
        color = wlen_to_rgb(key)
        if mode == 'residues':
            result = robust_fit(tdecay, idecay, model='double_neg')
            ydata = result.residual
            ax.plot(1E3*tdecay, ydata, color=color,
                marker='o', markersize=3., label=label)
        else:
            ydata = idecay
        ax.plot(1E3*tdecay, ydata, color=color,
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
    wavelength_list : typenormalize_spectrum
        Description of parameter `wavelength_list`.
    filename : type
        Description of parameter `filename`.

    Returns
    -------
    type
        Description of returned object.

    """
    fig, axes = plt.subplots(len(wlen_list), 1, figsize=[8, 6],
                             sharex=True, sharey=False)
    # ax_common = fig.add_subplot(111)    # The big subplot
    # ax_common.set_ylabel('Emission power (log(), U.A.)')

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
        annotate(ax, '$\\alpha_1$ = %.2f' % lp_params['m'], xytext=(.9, .45))
        annotate(ax, '$\\alpha_2$ = %.2f' % hp_params['m'], xytext=(.9, .25))
        # ax.legend(bbox_to_anchor=(0.35, 0.15), loc=2, borderaxespad=0.)
        # ax.set_xlim(-30, -6)
        ax.set_ylim(np.min(y)-5, np.max(y)+5)
        ax.set_xlabel('Log Excitation ($mW/cm^2$)')
    plt.sca(ax)
    p_to_dens = (4.35/0.008)**2/1000
    print(power_labels[0]*p_to_dens)
    xvalues = x[::5]
    labels = ['%.1f' % (float(p)*p_to_dens) for p in power_labels]
    labels = labels[::5]
    plt.xticks(xvalues, labels)
    # ax.tick_params(axis='x', pad=-30, labelcolor='k', bottom=False)

    axes[0].set_ylabel('Emitting power (log(), U.A.)')
    if filename:
        plt.savefig(filename)
    plt.show()

#
# def plot_3d(spectrum_list, wavelength_list, ax=None, filename=None):
#
#     # fig = plt.figure(figsize=[12,12])
#     ax = fig.gca(projection='3d')
#     verts = []
#     zs = [s['label'] for s in spectrum_list]
#     colors = [cm.hot(p/5000.) for p in zs]
#     for spectrum_dict in spectrum_list:
#     #     ys = np.random.rand(len(x))
#         x = spectrum_dict['x']
#         y = spectrum_dict['y']
#         # x = spec_dict[z][0]
#         # y = spec_dict[z][1]
#         y[0], y[-1] = 0, 0
#         verts.append(list(zip(x, y)))
#
#     poly = PolyCollection(verts,  facecolors=colors)
#     poly.set_alpha(1.)
#     ax.add_collection3d(poly, zs=zs, zdir='y')
#
#     ax.set_xlabel('Long. de onda (nm)')
#     ax.set_xlim3d(350, 700)
#     ax.set_ylabel('Potencia incidente (U.A.)')
#     ax.set_ylim3d(0, 5000)
#     ax.set_zlabel('Potencia de salida (U.A.)')
#     ax.set_zlim3d(0, 5.5E7)
#     if filename:
#         plt.savefig(filename, dpi=300)
#     plt.show()

def plot_spectrums(spectrum_list, filename=None, ax=None, normalize=True):

    import matplotlib.cm as cm
    if ax is None:
        ax = plt.gca()
    colors = iter([cm.afmhot(p) for p in np.linspace(0.05, 0.5, len(spectrum_list))])
    for spectrum_dict in spectrum_list:
        x = spectrum_dict['x']
        y = spectrum_dict['y']
        if normalize:
            y /= np.max(y)
        ax.plot(x, y, color=next(colors), linewidth=0.8)

    if filename:
        plt.savefig(filename)

def plot_efficiencies(spectrum_list, band_centers=[400, 530, 660], ax=None):
    import matplotlib.cm as cm
    if ax is None:
        ax = plt.gca()
    colors = iter([cm.afmhot(p) for p in np.linspace(0.05, 0.6, len(spectrum_list))])

    for spectrum_dict in spectrum_list:
        for bc in band_centers:
            pind = int(np.where(x == peak)[0])
            peak_amp = simps(y[pind-5:pind+5])
            peak_amps[peak].append(peak_amp)

    # for spectrum_dict in spectrum_list:
    #     x = spectrum_dict['x']
    #     y = spectrum_dict['y']
    #     y /= np.max(y)
    #     ax.plot(x, y, color=next(colors), linewidth=0.8)

    if filename:
        plt.savefig(filename)


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

def plot_3d(spectrum_list, wavelength_list, ax=None, filename=None):
    import matplotlib.cm as cm
    if ax is None:
        ax = plt.gca(projection='3d')
    ax = plt.gca(projection='3d')
    verts = []
    zs = [s['label'] for s in spectrum_list]
    colors = [cm.hot(p/8000.) for p in zs]
    for spectrum_dict in spectrum_list:
        x = spectrum_dict['x']
        y = spectrum_dict['y']
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

# def plot_3d(spectrum_list, wavelength_list, ax=None, filename=None):
#     # from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
#     # fig = plt.figure()
#     #
#     # ax = Axes3D(fig) #<-- Note the difference from your original code...
#     #
#     # X, Y, Z = axes3d.get_test_data(0.05)
#     # cset = ax.contour(X, Y, Z, 16, extend3d=True)
#     # ax.clabel(cset, fontsize=9, inline=1)
#     import matplotlib
#     print(matplotlib.projections.get_projection_names())
#

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
    wlens_iter = iter(wlen_list)
    labels = ['DE' for wl in wlen_list]

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
    xvalues = np.arange(len(wlen_list))
    bar_width = 0.8
    if var == 'f':
        f_array = np.array(f_list)
        bp = ax.bar(xvalues, f_array, bar_width)
        mono_ind = np.where(f_array>0.98)[0]
        for i in mono_ind:
            labels[int(i)] = 'ME'
    elif var == 'ka':
        bp = ax.bar(xvalues, ka_list, bar_width)
    elif var == 'tau_a':
        tau_a = [1/float(k) for k in ka_list]
        bp = ax.bar(xvalues, tau_a, bar_width)
    elif var == 'kUC':
        bp = ax.bar(xvalues, kUC_list, bar_width)
    elif var == 'tau_ETU':
        tau_ETU = [1/float(k) for k in kUC_list]
        bp = ax.bar(xvalues, tau_ETU, bar_width)
    for i in range(len(bp)):
        wlen = next(wlens_iter)
        bp[i].set_facecolor(wlen_to_rgb(wlen))
        bp[i].set_edgecolor('k')
    plt.sca(ax)
    plt.xticks(xvalues+0.05, labels, rotation=90)
    plt.yticks((0.0, 0.1, 0.2, 0.3))
    ax.tick_params(axis='x', pad=-30, labelcolor='k', bottom=False)
    return bp


def plot_taus(fit_dict, axes=None):
    if axes is None:
        ax = plt.gca()
    f_list = list()
    kUC_list = list()
    ka_list = list()
    tau_ETU = list()

    wlen_list = [float(w) for w in fit_dict.keys()]
    wlens_iter = iter(wlen_list)
    labels = ['%i' % wl for wl in wlen_list]

    for key, result in fit_dict.items():
        if(result.model.name == 'double_exponential'):
            a_ka = result.params['a1'].value
            a_kuc = result.params['a2'].value
            f = a_ka/(a_ka+a_kuc)
            f_list.append(a_ka/(a_ka+a_kuc))
            if f < 0.95:
                kuc = result.params['kUC'].value/1000
            else:
                kuc = 0
            kUC_list.append(kuc)
            ka = result.params['ka'].value/1000
            ka_list.append(ka)
        elif(result.model.name == 'exponential'):
            a_ka = result.params['a1'].value
            f = 1
            f_list.append(f)
            ka = result.params['ka'].value/1000
            ka_list.append(ka)
        print('Fitted parameters for:  %.2f nm' %  key)
        print(result.fit_report())
    xvalues = np.arange(len(wlen_list))
    bar_width = 0.8

    tau_a = [1/float(k) for k in ka_list]
    bpa = axes[0].bar(xvalues, tau_a, bar_width)
    for k in kUC_list:
        if k != 0:
            tau_ETU.append(1/float(k))
        else:
            tau_ETU.append(0)
    bpb = axes[1].bar(xvalues, tau_ETU, bar_width)
    for ax in axes:
        wlens_iter = iter(wlen_list)
        for i in range(len(bpa)):
            wlen = next(wlens_iter)
            bpa[i].set_facecolor(wlen_to_rgb(wlen))
            bpa[i].set_edgecolor('k')
            bpb[i].set_facecolor(wlen_to_rgb(wlen))
            bpb[i].set_edgecolor('k')
            plt.sca(ax)
            plt.xticks(xvalues+0.05, labels, rotation=90)
            # plt.yticks((0.0, 0.1, 0.2))
            ax.tick_params(axis='x', pad=-30, labelcolor='k', bottom=False)
    return

def plot_mean_taus(tdecay, mtime_dict, labels=None, ax=None):
    if ax is None:
        ax = plt.gca()
    taus = np.array(list(mtime_dict.values()))
    wlens = np.array(list(mtime_dict.keys()))
    xvalues = np.arange(len(wlens))
    bar_width = .8
    taus_ms = 1E3*taus
    barlist = ax.bar(xvalues, taus_ms, bar_width, edgecolor='k')
    wlens_iter = iter(wlens)
    if labels:
        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xvalues:
                return labels[int(tick_val)]
            else:
                return ''
        # Top ticks
        axT = ax.twiny()
        axT.tick_params(direction = 'in')
        plt.sca(axT)
        wlens = np.array(labels)
        xticks_peaks = (wlens-365)/315
        plt.xticks(xticks_peaks)
        ax.set_xlim([-0.9, 18])
        timeax_lims = ax.get_xlim()
        bars_postions = np.linspace(0+0.048, 1-0.058, len(xvalues))
        for tstart, tend, dy, wlen in zip(xticks_peaks, bars_postions, taus_ms, wlens):
            dx = tend-tstart
            # dy = -np.sqrt(0.1-dx**2)
            dy = -(0.4-dy)
            print(tstart, tend, dx, dy)
            linestyle = (0, (3, 10, 1, 10))
            axT.arrow(tstart, 0.4, dx, dy, head_width=0.0, fc=None, ec=wlen_to_rgb(wlen),
            linestyle='dotted')
        axT.tick_params(labeltop=False)
        # Bottom ticks
        plt.sca(ax)
        plt.xticks(xvalues+0.05, labels, rotation=90)
        plt.yticks((0.0, 0.1, 0.2, 0.3))
    ax.tick_params(axis='x', pad=-30, labelcolor='k', bottom=False)
    # ax.set_xticks(xvalues, rotation=90)
    # ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.set_xticks(ind, tuple(labels))
    for i in range(len(barlist)):
        wlen = next(wlens_iter)
        barlist[i].set_facecolor(wlen_to_rgb(wlen))
        barlist[i].set_edgecolor('k')
    return xvalues, barlist


def plot_spectrum(lambdas=None, spectrum=None, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    def format_fn(tick_val, tick_pos):
        if int(tick_val) in lambdas:
            return labels[int(tick_val)]
        else:
            return ''

    for name, value in kwargs.items():
        if name is 'framed':
            ax.set_frame_on(value)
            if value is False:
                plt.sca(ax)
                plt.xticks([])
                plt.yticks([])
        if name is 'normalized':
            if value is True:
                spectrum /= max(spectrum)
                ax.set_ylim([0, 1])
        if name is 'xlim':
            ax.set_xlim(value)

        ax.plot(lambdas, spectrum)

def plot_log_power(power_list, peak_amps, wlen, ax=None):
    if ax is None:
        ax = plt.gca()
    power_mw = np.array(power_list)
    log_power = np.log10(power_mw)
    log_amps = np.log10(peak_amps[wlen])
    # log_amps -= log_amps[-1]
    print('For %s' % wlen, peak_amps[wlen])
    ax.plot(log_power, log_amps, color=wlen_to_rgb(wlen),
            marker='o', markersize=1.2)

def add_linear_fitting(lambdas, spectrum_list, ax=None):

    import matplotlib.cm as cm
    if ax is None:
        ax = plt.gca()
    for spectrum_dict in spectrum_list:
        x = spectrum_dict['x']
        y = spectrum_dict['y']
        y /= np.max(y)
        ax.plot(x, y, color=next(colors))

    if filename:
        plt.savefig(filename)
