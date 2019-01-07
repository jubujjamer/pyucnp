# -*- coding: utf-8 -*-
"""
file datafit.py
author: Juan Marco Bujjamer

Data fitting for UCNP curves.
"""
import os
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.stats import chisquare
import matplotlib.cm as cm


from pyucnp import dynamics


# kFT es menor que 1E-17
Ner = 1E20
kFT = 1E-17
Wetu = kFT*Ner
