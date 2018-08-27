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
import itertools as it
from scipy.stats import chisquare


parslist = ['WA',  'WB', 'WC', 'WE', 'kFT', 'kB', 'kC', 'sS', 'sB', 'sC',
           'WCB', 'WDB', 'WDC', 'WEB', 'WEC', 'WED', 'WD']
zeroslist = ['A0', 'B0', 'C0', 'D0', 'E0', 'R0', 'S0']


def ucsystem(w, t, parameters):
    """
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

def ucsystem_simple(w, t, parameters):
    """
    Defines the differential equations for the coupled spring-mass system. Se
    ODE_analysis notebook for the definition of the model's parameters.

    Args:
    -----
        w (list):     list with the initial variable states
                      w = [N0, N1, N2 ]
        t (array):    time
        pars (list):  vector of the parameters:
                        p = [G, k1, k2a, wetu, E, k2b ]

    Returns:
    --------

    """
    N0, N1, N2 = w
    G, k1, k2a, wetu, E, k2b = parameters
    # Create f = (N0, N1, N2):
    f = [-G*N0+k1*N1+k2a*N2+wetu*N1**2,
         G*N0-E*N1-k1*N1+k2b*N2-2*wetu*N1**2,
         E*N1 - (k2a+k2b)*N2 + wetu*N1**2]
    return f

def ucsystem_simple_varpar(w, t, fixed_pars, G_fun):
    """
    Defines the differential equations for the coupled spring-mass system. Se
    ODE_analysis notebook for the definition of the model's parameters.

    Args:
    -----
        w (list):     list with the initial variable states
                      w = [N0, N1, N2 ]
        t (array):    time
        pars (list):  vector of the parameters:
                        p = [G, k1, k2a, wetu, E, k2b ]

    Returns:
    --------

    """
    N0, N1, N2 = w
    k1, k2a, wetu, E, k2b = fixed_pars
    # Create f = (N0, N1, N2):
    f = [-G_fun(t)*N0+k1*N1+k2a*N2+wetu*N1**2,
         G_fun(t)*N0-E*N1-k1*N1+k2b*N2-2*wetu*N1**2,
         E*N1-(k2a+k2b)*N2 + wetu*N1**2]
    return f

def odese(pars, tdata, ydata, stoptime, state):
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

    z0 = pars[18:]
    p0 = pars[:18]
    numpoints = len(tdata)
    # ODE solver parameters
    abserr, relerr = 1.0e-8, 1.0e-6
    # Create the time samples for the output of the import sympy as sp
    t = [stoptime*float(i)/(numpoints - 1) for i in range(numpoints)]

    # Call the ODE solver.
    wsol = odeint(ucsystem, z0, t, args=(p0,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    fsignal /= max(fsignal)

    error = sum((fsignal-ydata)**2)
    return t, fsignal, error


def simulate_system(tdata, ydata, state):
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
    stoptime = tdata[-1]

    parn = {'r': 0, 'kFT': 1, 'kB':2, 'kC': 3, 'sS': 4, 'sB': 5, 'sC': 6,
            'WA': 7, 'WB': 8, 'WC': 9, 'WD': 10, 'WE': 11, 'WCB': 12,
            'WDB': 13, 'WDC': 14, 'WEB': 15, 'WEC': 16, 'WED': 17,
            'A0': 18, 'B0': 19, 'C0': 20, 'D0': 21, 'E0': 22, 'R0': 23, 'S0': 24}


    # Initial conditions
    # z0 = [1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E5]  # w0 = [A0, B0, C0, D0, E0, R0, S0]
    [A0, B0, C0, D0, E0, R0, S0] = [1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E5]
    # Laser power
    r = 0.
    # k coeffs:
    kFT, kB, kC = 0, 0, 1
    # Cross sections
    sS, sB, sC = 0, 0, 0
    # Decay rates
    WA, WB, WC, WD, WE = 1E3, 1E3, 1E1, 1E3, 1E4
    WCB, WDB, WDC, WEB, WEC, WED = 1E3, 1E3, 1E3, 1E3, 1E3, 1E3
    p0 = [r, kFT, kB, kC, sS, sB, sC, WA, WB, WC, WD, WE, WCB, WDB, WDC, WEB, WEC, WED, A0, B0, C0, D0, E0, R0, S0]

    z0 = p0[18:]
    p0 = p0[:18]

    numpoints = len(tdata)
    abserr, relerr = 1.0e-5, 1.0e-5
    # Create the time samples for the output of the import sympy as sp
    # t = [stoptime*float(i)/(numpoints - 1) for i in range(numpoints)]
    t = np.linspace(0, stoptime, numpoints)

    wsol = odeint(ucsystem, z0, t, args=(p0,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    # fsignal /= max(fsignal)
    return t, fsignal


def simulate_simple_system(tdata, ydata, state):
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
    statedict = {'N0': 0, 'N1': 1, 'N2': 2}
    stoptime = tdata[-1]
    parn = {'G': 0, 'k1': 1, 'k2a': 2, 'k2b': 3, 'E': 4, 'wetu': 5}
    # Initial conditions
    # [N00, N10, N20] = [2E20, 2, 1.2E-19]
    [N00, N10, N20] = [2E20, 2,  8E-20]
    z0 = [N00, N10, N20]
    # Laser power
    G, k1, k2a, wetu, E, k2b = [0, 1, 20, 0, 0, 20]
    p0 = [G, k1, k2a, wetu, E, k2b]
    abserr, relerr = 1.0e-5, 1.0e-5
    # Create the time samples for the output of the import sympy as sp
    t = np.linspace(0, stoptime, len(tdata))
    wsol = odeint(ucsystem_simple, z0, t, args=(p0,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    # fsignal /= max(fsignal)
    return t, fsignal

def simulate_simple_input(tdata, ydata, state):
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
    statedict = {'N0': 0, 'N1': 1, 'N2': 2}
    stoptime = tdata[-1]
    starttime = tdata[0]
    # parn = {'G': 0, 'k1': 1, 'k2a': 2, 'k2b': 3, 'E': 4, 'wetu': 5}
    abserr, relerr = 1.0e-8, 1.0e-8
    # Initial conditions
    # [N00, N10, N20] = [2E20, 2, 1.2E-19]
    [N00, N10, N20] = [2E20, 0,  0]
    z0 = [N00, N10, N20]
    # Laser power
    N = 10000
    t = np.linspace(starttime, stoptime, N)

    def G_fun(t):
        if t <= 0:
            ex = 1E-18
            return ex
        else:
            return 0

    # k1, k2a, wetu, E, k2b = [5, 2, 0.005, .020, 2]
    k1, k2a, wetu, E, k2b = [5, 2, 5, .020, 2]
    fixed_p = [k1, k2a, wetu, E, k2b]

    p0 = [k1, k2a, wetu, E, k2b]
    # Create the time samples for the output of the import sympy as sp
    wsol = odeint(ucsystem_simple_varpar, z0, t, args=(fixed_p, G_fun), atol=abserr, rtol=relerr)
    fsignal = wsol.T[statedict[state]]
    return t, wsol.T



def odefitse(fitic, p0, fit_pars, tdata, ydata, stoptime, state):
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
    for par in fit_pars:
        p0[par] = next(fitic_iter)
    z0 = p0[18:]
    p0 = p0[:18]

    numpoints = len(tdata)
    abserr, relerr = 1.0e-5, 1.0e-5
    # Create the time samples for the output of the import sympy as sp
    t = [stoptime*float(i)/(numpoints - 1) for i in range(numpoints)]
    wsol = odeint(ucsystem, z0, t, args=(p0,), atol=abserr, rtol=relerr)
    fsignal = np.array([s[statedict[state]] for s in wsol])
    # Normalize amplitude
    fsignal /= max(fsignal)
    sqe = sum((fsignal-ydata)**2)
    return sqe


def odefitpars(fitpars, p0, tdata, ydata, stoptime, state):
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
    parn = {'r': 0, 'kFT': 1, 'kB':2, 'kC': 3, 'sS': 4, 'sB': 5, 'sC': 6,
            'WA': 7, 'WB': 8, 'WC': 9, 'WD': 10, 'WE': 11, 'WCB': 12,
            'WDB': 13, 'WDC': 14, 'WEB': 15, 'WEC': 16, 'WED': 17,
            'A0': 18, 'B0': 19, 'C0': 20, 'D0': 21, 'E0': 22, 'R0': 23, 'S0': 24}

    fitic = [p0[parn[par]] for par in fitpars]
    fitindex = [parn[par] for par in fitpars]

    parameter = fitpars[0]
    if parameter in parslist:
        result = scipy.optimize.least_squares(odefitse, x0=fitic,
                    bounds=(0, 1E5), ftol=1e-08, xtol=1e-08, gtol=1e-08,
                    args=(p0, fitindex, tdata, ydata, stoptime, state))
    elif parameter in zeroslist:
        result = scipy.optimize.least_squares(odefitse, x0=fitic,
                    bounds=(0, 1E5), ftol=1e-08, xtol=1e-08, gtol=1e-08,
                    args=(p0, fitindex, tdata, ydata, stoptime, state))


    fititer = iter(fitindex)
    for pf in result['x']:
        p0[next(fititer)] = pf

    return p0, result



# def odefitzeros(fitzeros, p0, tdata, ydata, z0, stoptime, state):
#     """ LS error estimator between the data and an estimated model using the p
#     parameters (see ucsistem()).
#
#     Args:
#     -----
#         fitpars: (list) list of parameter to be fit. If a parameter is not
#                         needed to be fit put None on its place.
#         tdata: (array) the time array of the experimental data.
#         ydata: (array) the experimental data.
#         pars:  (list)  the parameters list
#                        [S0, r, kFT, kB1E2, kC, sS, sB, sC,
#                         WA, WB, WC, WD, WE, WCB, WDB, WEB, WEC, WED]
#
#     Returns:
#     --------
#         (float) squared error
#     """
#     zerosn = {'A0': 0, 'B0': 1, 'C0': 2, 'D0': 3, 'E0': 4, 'R0': 5, 'S0': 6}
#
#     fitic = [z0[zerosn[par]] for par in zerosn]
#     fitindex = [zerosn[par] for par in zerosn]
#     result = scipy.optimize.least_squares(odefitse, x0=fitic,
#                 bounds=(0, 1e5), ftol=1e-05, xtol=1e-03, gtol=1e-05,
#                 args=(p0, fitindex, tdata, ydata, z0, stoptime, state, 'zeros'))
#     fititer = iter(fitindex)
#     for pf in result['x']:
#         z0[fititer.next()] = pf
#     # if result['nfev'] > 20:
#     #     print(fitzeros)
#     return z0, result


def ucfit(tdata, ydata, fitlist, abserr=1E-2, curve=None):
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
    parlist = ['WA',  'WB', 'WC', 'WE', 'kFT', 'kB', 'kC', 'sS', 'sB', 'sC',
               'WCB', 'WDB', 'WDC', 'WEB', 'WEC', 'WED', 'WD',
               'A0', 'B0', 'C0', 'D0', 'E0', 'R0', 'S0']
    stoptime = tdata[-1]

    # Initial conditions
    # z0 = [1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E5]  # w0 = [A0, B0, C0, D0, E0, R0, S0]
    [A0, B0, C0, D0, E0, R0, S0] = [1E4, 1E4, 1E4, 1E4, 1E4, 1E4, 1E5]
    # Laser power
    r = 0.
    # k coeffs:
    kFT, kB, kC = 0, 0, 1
    # Cross sections
    sS, sB, sC = 0, 0, 0
    # Decay rates
    WA, WB, WC, WD, WE = 1E3, 1E3, 1E1, .1E4, 1
    WCB, WDB, WDC, WEB, WEC, WED = 1E3, 1E3, 1E3, 1E3, 1E3, 1E3
    p0 = [r, kFT, kB, kC, sS, sB, sC,
          WA, WB, WC, WD, WE, WCB, WDB, WDC, WEB, WEC, WED,
          A0, B0, C0, D0, E0, R0, S0]

    # Initial search of parameters
    pariter = it.cycle(fitlist)
    prev_cost = 0
    weights = dict()
    prev_weights = dict()
    fit_dict = dict()
    for e in fitlist:
        weights[e] = 0
        prev_weights[e] = 0
        fit_dict[e] = 0

    maxiter = 100
    niter = 0
    for par in pariter:
        fitpars = [par]
        pfit, result = odefitpars(fitpars, p0, tdata, ydata, stoptime, curve)
        weights[par] += np.abs(prev_weights[par]-result['cost'])
        prev_weights[par] = result['cost']
        cost_sum = sum(prev_weights.values())
        niter += 1
        if result['cost'] < abserr or niter > maxiter:
            break

    t, fsignal, error = odese(pfit, tdata, ydata, stoptime, curve)
    ji2, p = scipy.stats.chisquare((ydata-fsignal)**2, ddof=10)
    print(ji2, result['cost'])


    for name, val in zip(parlist, pfit):
        fit_dict[name] = val
    return t, fsignal, weights, fit_dict


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
