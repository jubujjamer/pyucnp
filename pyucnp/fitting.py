# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 14:34:04 2016
@author: Juan
"""
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import csv
import matplotlib.pyplot as plt

# Task definition taken form NIMAX*********
folder = '../samples_NI/2016_0108/'
fname = 'F_003_001.csv';
fout = folder+fname;
NP = 1000; # Sampling points
FS = 300.E3; # Sampling frequency
# *****************************************

def fit_exponential(x,, y):
    return


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
