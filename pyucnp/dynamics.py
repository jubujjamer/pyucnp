#!/usr/bin/python
# -*- coding: utf-8 -*-
""" geometry.py
Last update: 11/07/2017

Dynamical process simulation.
Usage:

"""
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.integrate import odeint
from sympy import symbols, cos, sin
import random



def ucsystem(w, t, parameters):
    """
    Defines the differential equations for the coupled spring-mass system. Se
    ODE_analysis notebook for the definition of the model's parameters.

    Args:
    -----
        w (list):     list with the initial variable states
                      w = [A0, B0, C0, D0, E0, R0, S0]
        t (array):    time
        pars (list):  vector of the parameters:
                        p = [S0, r, kFT, kB, kC, sS, sB, sC,
                             WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------

    """
    A, B, C, D, E, R, S = w
    r, kFT, kB, kC, sS, sB, sC, WA, WB, WC, WD, WE, WCB, WDB, WDC, WEB, WEC, WED = parameters

    # Create f = (A', B', C', D', E', R', S'):
    f = [0,
         -r*sB*B - kB*B*S - WB*B + WCB*C + WDB*D + WEB*E,
         kFT*A*S - r*sC*C - kC*C*S - WC*C + WDC*D + WEC*E,
         r*sB*B + kB*B*S - WD*D + WED*E,
         r*sC*C + kC*C*S - WE*E,
         0,
         r*sS*R - kFT*A*S - kB*B*S - kC*C*S]
    return f


def odese(pars, tdata, ydata, w0, stoptime, state):
    """ LS error estimator between the data and an estimated model using the p
    parameters (see ucsistem()).

    Args:
    -----
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [S0, r, kFT, kB, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------
        (float) squared error
    """
    statedict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'R': 5, 'S': 6}

    numpoints = len(tdata)
    # ODE solver parameters
    abserr, relerr = 1.0e-8, 1.0e-6
    # Create the time samples for the output of the import sympy as sp
    t = [stoptime*float(i)/(numpoints - 1) for i in range(numpoints)]

    # Call the ODE solver.
    wsol = odeint(ucsystem, w0, t, args=(pars,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    fsignal /= max(fsignal)

    error = sum((fsignal-ydata)**2)
    return t, fsignal, error



def odefitse(fitic, p0, fit_pars, tdata, ydata, z0, stoptime, state, mode='pars'):
    """ LS error estimator between the data and an estimated model using the p
    parameters (see ucsistem()).

    Args:
    -----
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [S0, r, kFT, kB, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------
        (float) squared error
    """
    statedict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'R': 5, 'S': 6}

    fitic_iter = iter(fitic)
    if mode=='pars':
        for par in fit_pars:
            p0[par] = fitic_iter.next()
    elif mode=='zeros':
        for z in fit_pars:
            z0[z] = fitic_iter.next()
    numpoints = len(tdata)
    abserr, relerr = 1.0e-5, 1.0e-5
    # Create the time samples for the output of the import sympy as sp
    t = [stoptime*float(i)/(numpoints - 1) for i in range(numpoints)]
    wsol = odeint(ucsystem, z0, t, args=(p0,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    fsignal /= max(fsignal)
    error = sum((fsignal-ydata)**2)
    return error


def odefitpars(fitpars, p0, tdata, ydata, w0, stoptime, state):
    """ LS error estimator between the data and an estimated model using the p
    parameters (see ucsistem()).

    Args:kCCS
    -----
        fitpars: (list) list of parameter to be fit. If a parameter is not
                        needed to be fit put None on its place.
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [S0, r, kFT, kB, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------
        (float) squared error
    """
    parn = {'r': 0, 'kFT' :1, 'kB' :2, 'kC': 3, 'sS': 4, 'sB': 5, 'sC': 6,
            'WA': 7, 'WB': 8, 'WC' : 9, 'WD': 10, 'WE': 11, 'WCB': 12,
            'WDB': 13, 'WDC': 14, 'WEB': 15, 'WEC': 16, 'WED': 17}

    fitic = [p0[parn[par]] for par in fitpars]
    fitindex = [parn[par] for par in fitpars]
    result = scipy.optimize.least_squares(odefitse, x0=fitic,
                bounds=(0, 1E5), ftol=1e-08, xtol=1e-08, gtol=1e-08,
                args=(p0, fitindex, tdata, ydata, w0, stoptime, state))
    fititer = iter(fitindex)
    for pf in result['x']:
        p0[fititer.next()] = pf
    # print(result)
    # if result['nfev'] > 18:
    #     print(fitpars)
    # print(result)

    return p0, result



def odefitzeros(fitzeros, p0, tdata, ydata, z0, stoptime, state):
    """ LS error estimator between the data and an estimated model using the p
    parameters (see ucsistem()).

    Args:
    -----
        fitpars: (list) list of parameter to be fit. If a parameter is not
                        needed to be fit put None on its place.
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [S0, r, kFT, kB1E2, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------
        (float) squared error
    """
    zerosn = {'A0': 0, 'B0': 1, 'C0': 2, 'D0': 3, 'E0': 4, 'R0': 5, 'S0': 6}

    fitic = [z0[zerosn[par]] for par in zerosn]
    fitindex = [zerosn[par] for par in zerosn]
    result = scipy.optimize.least_squares(odefitse, x0=fitic,
                bounds=(0, 1e5), ftol=1e-05, xtol=1e-03, gtol=1e-05,
                args=(p0, fitindex, tdata, ydata, z0, stoptime, state, 'zeros'))
    fititer = iter(fitindex)
    for pf in result['x']:
        z0[fititer.next()] = pf
    # if result['nfev'] > 20:
    #     print(fitzeros)
    return z0, result


def ucfit(tdata, ydata):
    """ UC fit using the measured data and the dynamical model.

    Args:
    -----
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [r, kFT, kB, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------        random.shuffl

        (float) LS error
    """
    # parlist = ['r', 'kFT', 'kB', 'kC', 'sS', 'sB', 'sC',
    #            'WA', 'WB', 'WC', 'WD', 'WE',
    #            'WCB', 'WDB', 'WDC', 'WEB', 'WEC', 'WED']
    zeroslist = ['A0', 'B0', 'C0', 'D0', 'E0', 'R0', 'S0']
    parlist = ['WA',  'WB', 'WC', 'WE', 'kFT', 'kB', 'kC', 'sS', 'sB', 'sC',
               'WCB', 'WDB', 'WDC', 'WEB', 'WEC', 'WED', 'WD']
    fitlist = zeroslist+parlist

    stoptime = tdata[-1]
    # Initial conditions
    z0 = [1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E5]  # w0 = [A0, B0, C0, D0, E0, R0, S0]
    # Laser power
    r = 0.
    # k coeffs:
    kFT, kB, kC = 0, 0, 1
    # Cross sections
    sS, sB, sC = 0, 0, 0
    # Decay rates
    WA, WB, WC, WD, WE = 1E3, 1E3, 1E1, .1E4, 1E4
    WCB, WDB, WDC, WEB, WEC, WED = 1E3, 1E3, 1E3, 1E3, 1E3, 1E3
    # Pack up the parameters and initial con + WDC*D + WEC*E
    # for WD in np.linspace(.1E4, .9E4, 100):
    # Calculate squared error
    p0 = [r, kFT, kB, kC, sS, sB, sC, WA, WB, WC, WD, WE, WCB, WDB, WDC, WEB, WEC, WED]


    # Initial search of parameters
    prev_cost = 0
    for i in range(10):
        delta_cost_list = list()
        for par in fitlist:
            if par in zeroslist:
                fitzeros = [par]
                zfit, result = odefitzeros(fitzeros, p0, tdata, ydata, z0, stoptime, 'E')
            else:
                fitpars = [par]
                pfit, result = odefitpars(fitpars, p0, tdata, ydata, zfit, stoptime, 'E')
            delta_cost_list.append([np.abs(prev_cost-result['cost']), par])
            prev_cost = result['cost']
            print(prev_cost)
        random.shuffle(fitlist)
        print(i)
    fitlist = [s[1] for s in sorted(delta_cost_list, reverse=True) if s[0]>1E-8]
    print(fitlist)
    # Final convergence
    prev_cost = 0
    for i in range(10):
        delta_cost_list = list()
        for par in fitlist:
            if par in zeroslist:
                fitzeros = [par]
                zfit, result = odefitzeros(fitzeros, p0, tdata, ydata, z0, stoptime, 'E')
            else:
                fitpars = [par]
                pfit, result = odefitpars(fitpars, p0, tdata, ydata, zfit, stoptime, 'E')
            delta_cost_list.append([np.abs(prev_cost-result['cost']), par])
            prev_cost = result['cost']
            # fitlist = [s[1] for s in sorted(delta_cost_list, reverse=True)]
        random.shuffle(fitlist)
        # if i>2:
        #     fitlist = [s[1] for s in sorted(delta_cost_list, reverse=True) if s[0]>1E-8]
        print(fitlist, prev_cost)
        # random.shuffle(zeroslist)
        # random.shuffle(fitlist)
    t, fsignal, error = odese(pfit, tdata, ydata, zfit, stoptime, 'E')
    print(error)
        # print(zeroslist)

    # for name, value in zip(zeroslist, zfit):
    #     print(name, value)
    # for name, value in zip(fitlist, pfit):
    #     print(name, value)
    # print result


    return t, fsignal, delta_cost_list, pfit


def plot_solution(tdata, ydata):
    """ UC fit using the measured data and the dynamical model.

    Args:
    -----
        tdata: (array) the time array of the experimental data.
        ydata: (array) the experimental data.
        pars:  (list)  the parameters list
                       [S0, r, kFT, kB, kC, sS, sB, sC,
                        WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]

    Returns:
    --------
        (float) LS error
    """
    # Initial conditions
    w0 = [10, 0, 0, 0, 0, 2, 0]  # w0 = [A0, B0, C0, D0, E0, R0, S0]
    # Laser power
    r = .001
    # k coeffs:
    kFT, kB, kC = 1, 2, 5
    # Cross sections
    sS, sB, sC = 1, .005, .1
    # Decay rates
    WA, WB, WC, WD, WE = .01, .1, 0.05, 2, .5
    WCB, WDB, WDC, WEB, WEC, WED = 1, 0, 0, 0, 0, 0
    # Pack up the parameters and initial con + WDC*D + WEC*E
    pars = [r, kFT, kB, kC, sS, sB, sC,
            WA, WB, WC, WD, WE, WCB, WDB, WDC, WEB, WEC, WED]
    wsol, error = ode_ls(tdata, ydata, pars, w0)  # Calculate ls error

    #with open('ucsystem.dat', 'w') as f:
        # Print & save the solution.
    w1 = zip(t, wsol)
    #    w1[0], w1[1], w1[2], w1[3], w1[4])
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=[14, 4])
    l_names = ['A', 'B', 'C', 'D', 'E', 'R', 'S']
    names = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'R', 6:'S'}
    signals = dict()
    i = 0
    for l in l_names:
        signals[l] = np.asarray([s[i] for s in wsol])
        i += 1
    ax.plot(t, signals['D'], label = 'Rojo (Er, 5)')
    ax.plot(t, signals['E'], label = 'Verde (Er, 6)')
    ax.plot(t, signals['D']+signals['E'], label = 'Suma (sin filtro)', lw = 2)
    #ax.set_ylim([0, 1]);
    ax.legend();
    ax.set_title('Simulacin de (Er, 5) y (Er, 6)');
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    plt.show()
    # print(ucsystem())

    return
