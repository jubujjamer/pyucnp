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


def fit_exponential(tdata, ydata, init_params=None, model='single'):

    def etu_esa_mixture(t, ka, kUC, a1, a2):
        return a1*np.exp(ka*t)+a2*(1-np.exp((ka+kUC)*t))*np.exp(-kUC*t)

    def exponential(t, ka, kUC, a1, a2):
        return a1 * np.exp(-t*ka)

    def double_exponential(t, ka, kUC, a1, a2):
        return a1 * np.exp(-t*ka) + a2 * np.exp(-t*kUC)

    def double_neg(t, ka, kUC, a1, a2):
        return a1 * np.exp(-t*ka) + a2 * np.exp(-t*kUC)

    def residual_single(p):
       return p['a1']*np.exp(-tdata/p['t1'])-ydata

    def residual_double(p):
       return p['a1']*np.exp(-tdata/p['t1']) + p['a2']*np.exp((tdata-0.1)/p['t2'])-ydata

    nbins = len(tdata)
    if model == 'single':
        model = lmfit.Model(exponential)
        if init_params is not None:
            a1, a2, kUC, ka = init_params
        model.set_param_hint('a1', value=a1, min=0, max=2)
        model.set_param_hint('a2', value=a2, min=0, max=1E-2)
        model.set_param_hint('kUC', value=kUC, min=0, max=10)
        model.set_param_hint('ka', value=ka, min=0, max=1e5)
        params = model.make_params()
    elif model == 'double':
        model = lmfit.Model(double_neg)
        if init_params is not None:
            a1, a2, kUC, ka = init_params
        model.set_param_hint('a1', value=a1, min=0, max=2)
        model.set_param_hint('a2', value=a2, min=0, max=2)
        model.set_param_hint('kUC', value=kUC, min=0, max=1e5)
        model.set_param_hint('ka', value=ka, min=0, max=1e5)
        params = model.make_params()
    elif model == 'double_neg':
        model = lmfit.Model(double_exponential)
        if init_params is not None:
            a1, a2, kUC, ka = init_params
        model.set_param_hint('a1', value=a1, min=0, max=2)
        model.set_param_hint('a2', value=a2, min=-1, max=0)
        model.set_param_hint('kUC', value=kUC, min=0, max=1e5)
        model.set_param_hint('ka', value=ka, min=0, max=1e5)
        params = model.make_params()
    elif model == 'mixture':
        model = lmfit.Model(etu_esa_mixture)
        if init_params is not None:
            a1, a2, kUC, ka = init_params
        model.set_param_hint('a1', value=a1, min=0, max=2)
        model.set_param_hint('a2', value=a2, min=0, max=3)
        model.set_param_hint('kUC', value=kUC, min=0, max=1e5)
        model.set_param_hint('ka', value=ka, min=-5e5, max=0)
        params = model.make_params()
    # r1 = model.fit(ydata, t=tdata, params=params, method='Nelder')
    r2 = model.fit(ydata, t=tdata, params=params, method='leastsq', nan_policy='omit')
    # out2 = mini.minimize(method='leastsq', params=out1.params)
    # lmfit.report_fit(out2.params, min_correl=0.5)
    # ci, trace = lmfit.conf_interval(mini, out2, sigmas=[1, 2],
    #                                 trace=True, verbose=False)
    # lmfit.printfuncs.report_ci(ci)
    # print(r2.ci_report())
    return r2


def robust_fit(tdata, ydata, init_params=None, model='single'):
    a1_list = np.arange(0, 1.2, .2)
    a2_list = np.arange(-1.2, 0, .2)
    kUC_list = np.arange(1E4, 1E5, 500E4)
    ka_list = np.arange(0, 1E4, 500E3)
    # ka_list = np.arange(0, -1E5, -500E3)
    init_iter = it.product(a1_list, a2_list, kUC_list, ka_list)
    chisq_min = np.inf
    r_min = None
    for init in init_iter:
        # init_params = [.8, .9, 3E3, -8E4]
        result = fit_exponential(tdata, ydata, init_params=init, model=model)
        if result.chisqr < chisq_min:
            chisq_min = result.chisqr
            r_min = result
    return r_min


def fit_line(x, y):
    def residual(params, x, data):
        b = params['b']
        m = params['m']
        model = b + m*x
        return (data-model)
    params = lmfit.Parameters()
    params.add('b', value=1.)
    params.add('m', value=1.)
    result = lmfit.minimize(residual, params, args=(x, y.reshape(1, len(y))))
    return result.params

def fit_power(x, y):
    def cuadratic(x, a0, a1, a2):
        return a0 + a1*x + a2*x**2
    model = lmfit.Model(cuadratic)
    model.set_param_hint('a0', value=1)
    model.set_param_hint('a1', value=1)
    model.set_param_hint('a2', value=1)
    params = model.make_params()
        # r1 = model.fit(ydata, t=tdata, params=params, method='Nelder')
    result = model.fit(y, x=x, params=params, method='leastsq', nan_policy='omit')
    parvals = [result.params[p].value for p in result.params]
    return result, parvals
