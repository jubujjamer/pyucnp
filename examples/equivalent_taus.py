# Check styles directory
import numpy as np
import matplotlib.pyplot as plt

from pyucnp.fitting import robust_best_fit
from pyucnp import data
import pyucnp.plotting as up
from pyucnp.experiment import Spectrum


measurement_day = '2019-02-06'
filename = 'sample_1.sp'
cfg = data.load_data(measurement_day, )
# Open measurements data
data.load_pickled(daystr, filename)
with open('filename', 'rb') as infile:
    sdata = pickle.load(infile)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[5, 4], sharex=True)
axiter = iter(axes)
