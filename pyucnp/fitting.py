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


def fit_exponential(tdata, ydata, model='single'):

    def etu_esa_mixture(t, ka, kUC, a1, a2):
        return a1*np.exp(ka*t)+a2*(1-np.exp((ka+kUC)*t))*np.exp(-kUC*t)

    def exponential(t, a1, t1):
        return a1 * np.exp(-(t/t1))

    def double_exponential(t, a1, t1, a2, t2):
        return a1 * np.exp(-(t/t1)) + a2 * np.exp(t/t2)

    def residual_single(p):
       return p['a1']*np.exp(-tdata/p['t1'])-ydata

    def residual_double(p):
       return p['a1']*np.exp(-tdata/p['t1']) + p['a2']*np.exp((tdata-0.1)/p['t2'])-ydata

    nbins = len(tdata)
    if model == 'single':
        model = lmfit.Model(exponential)
        params = model.make_params(a1=1, min=-10, max=10, t1=100E-6)
    elif model == 'double':
        model = lmfit.Model(double_exponential)
        model.set_param_hint('a1', value=ydata.max(), min=-10, max=10)
        model.set_param_hint('a2', value=.1, min=.1, max=10)
        model.set_param_hint('t1', value=350.E-6, min=0, max=1e-3)
        model.set_param_hint('t2', value=10.E-6, min=1e-4, max=1E-3)
        params = model.make_params()
    elif model == 'mixture':
        model = lmfit.Model(etu_esa_mixture)
        model.set_param_hint('a1', value=.8, min=0, max=2)
        model.set_param_hint('a2', value=.2, min=-1, max=3)
        model.set_param_hint('kUC', value=5E4, min=1e4, max=5e6)
        model.set_param_hint('ka', value=-1E2, min=-5e4, max=-10)
        params = model.make_params()
    # r1 = model.fit(ydata, t=tdata, params=params, method='Nelder')
    r2 = model.fit(ydata, t=tdata, params=params, method='leastsq')

    # out2 = mini.minimize(method='leastsq', params=out1.params)
    # lmfit.report_fit(out2.params, min_correl=0.5)
    # ci, trace = lmfit.conf_interval(mini, out2, sigmas=[1, 2],
    #                                 trace=True, verbose=False)
    # lmfit.printfuncs.report_ci(ci)
    # print(r2.ci_report())
    return r2


# ydata = np.log(-y-np.min(-y)+0.00001)
# x_cut = x[start:stop];
# y_cut = ydata[start:stop];
# x_cut = sm.add_constant(x_cut)
# print np.shape(x),np.shape(y)
# # plt.plot(x[:,1],y)
# # plt.show()
# # ydata = np.log(-y-0.0006)
#
#
# # Least squares estimation
# model = sm.OLS(y_cut, x_cut)
# results = model.fit()
# print(results.summary())
# print('Parameters: ', results.params)
# print('R2: ', results.rsquared)
#
#
# # We pick 100 hundred points equally spaced from the min to the max
# x_prime = np.linspace(x_cut[:,1].min(), x_cut[:,1].max(), 100)[:, np.newaxis]
# x_prime = sm.add_constant(x_prime)  # add constant as we did before
# # Now we calculate the predicted values
# y_hat = results.predict(x_prime)
# # Plots
# plt.scatter(x_cut[:,1]*TCONV, y_cut, alpha=0.3)  # Plot the raw data
# plt.plot(x_prime[:, 1]*TCONV, y_hat, 'r',linewidth=2.0, alpha=0.9)  # Add the regression line, colored in red
# plt.ylabel('Amplitud (V)')
# plt.xlabel('Tiempo (ms)')
# plt.title('Tiempo de vida con filtro Edmund')
# #plt.axis((0,3.5,-10,-3))
# plt.show()
