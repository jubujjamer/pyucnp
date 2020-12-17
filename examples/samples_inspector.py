# Check styles directory
import matplotlib
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
import pickle

from pyucnp.fitting import robust_best_fit
import pyucnp.data as dt
import pyucnp.plotting as up
from pyucnp.experiment import Spectrum
# plt.style.use('paper_ucnp')

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}

rc('font', **font)

# Open measurements data
cfg = dt.load_cfg(daystr='2017-09-22', config_file='analysis_dataset.yaml')
with open('filename', 'rb') as infile:
    sdata = pickle.load(infile)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=[5, 4], sharex=True)
axiter = iter(axes)

# Get alphas y decay parameters
wlen_list = sdata.relevant_peaks
alphas = np.zeros((len(wlen_list), 2))
taus = np.zeros((len(wlen_list), 2))
# Store mean values as [alpha1, alpha2, tau1, tau2]
xvalues = np.array([0, 1,  3, 4, 5, 6, 7, 8, 9,   11, 12, 13,   15, 16, 17, 18,   20, 21])
bands_indexes = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4])
mean_values = {'UVA':[0, 0 , 0 ,0] , 'B':[0, 0 , 0 ,0] , 'G':[0, 0 , 0 ,0] ,
             'Y':[0, 0 , 0 ,0] , 'R':[0, 0 , 0 ,0]}
# Calculate taus and alpha mean values
wlens, weights = sdata.intensityPeaks(index=33)
# Normalize weights with respect to integral
weights = weights/np.sum(weights)
for i, wlen in enumerate(sdata.relevant_peaks):
    alphas[i] = sdata.limitingSlopes(wlen)
    a1, tau1, a2, tau2 = sdata.decay_parameters(33, wlen)
    if a2/a1<0.16:
        tau2 = 0
    taus[i] = [tau1, tau2]

mean_alphas = list()
mean_tau1 = list()
mean_tau2 = list()
for index in range(5):
    mask = np.where(bands_indexes == index)
    masked_weights = weights[mask]
    alphas_mask = alphas[mask]
    mean_alphas.append(sum(alphas_mask)/len(alphas_mask))

    taus_mask = taus[mask]
    taus1 = [t[0] for t in taus_mask]
    taus2 = [t[1] for t in taus_mask]
    mean_tau1.append(sum(masked_weights*taus1)/sum(masked_weights))
    weights_tau2 = masked_weights[np.where(taus2 != 0)]
    mean_tau2.append(sum(masked_weights*taus2)/sum(masked_weights))
    # mean_tau2.append(sum(taus2)/max(len(taus2),1))
    print('mean tau 2 ',    mean_tau2)
# Generate barplots
# xvalues = np.arange(len(wlen_list))
bpa = axes[0].bar(xvalues, alphas[:, 0], width=0.8, )
bpb = axes[1].bar(xvalues, alphas[:, 1], width=0.8, )
bpc = axes[2].bar(xvalues, taus[:, 0], width=0.8, )
bpd = axes[3].bar(xvalues, taus[:, 1], width=0.8, )

wlens_iter = iter(wlen_list)
for i in range(len(bpa)):
    wlen = next(wlens_iter)
    bpa[i].set_facecolor(up.wlen_to_rgb(wlen))
    bpa[i].set_edgecolor('k')
    bpb[i].set_facecolor(up.wlen_to_rgb(wlen))
    bpb[i].set_edgecolor('k')
    bpc[i].set_facecolor(up.wlen_to_rgb(wlen))
    bpc[i].set_edgecolor('k')
    bpd[i].set_facecolor(up.wlen_to_rgb(wlen))
    bpd[i].set_edgecolor('k')

axes[0].set_ylabel('$\\alpha_1$', fontsize=12)
axes[1].set_ylabel('$\\alpha_2$', fontsize=12)
axes[2].set_ylabel('$\\tau_1$', fontsize=12)
axes[3].set_ylabel('$\\tau_2$', fontsize=12)
axes[3].set_xlabel('Wavelength', fontsize=12)
axes[0].set_ylim([0, 4])

axes[1].set_ylim([0, 2])
axes[2].set_ylim([0, 0.4])
axes[3].set_ylim([0, .1])

plt.sca(axes[3])
plt.xticks(xvalues+0.05, wlen_list, rotation=90)
axes[1].tick_params(axis='x', pad=-142, labelcolor='k', bottom=False)

## Time to add means for comparrison
## Peviously calculated means
mean_taus = {'UVA': ([379, 383], 0.1135, 0), 'B': ([410, 490], 0.1660, 0), 'G':
             ([504, 529], 0.1823, 0), 'Y': ([541, 557], 0.1977, 0.01537), 'R':
             ([654, 661], 0.3383, 0.07931)}
mean_alphas = {'UVA': ([379, 383], 2.7979, 1.5408), 'B': ([410, 490], 2.7929, 1.2875), 'G': ([504, 529], 1.9416, 1.1855), 'Y': ([541, 557], 1.8505, 0.9111), 'R': ([654, 661], 2.3482, 0.9685)}

for name, alphas_tuple in mean_alphas.items():
    print(alphas_tuple)
    col = 10
    band, alpha1, alpha2 = alphas_tuple
    index1 = xvalues[sdata.relevant_peaks.index(band[0])]
    index2 = xvalues[sdata.relevant_peaks.index(band[1])]
    axes[0].plot([index1-0.5, index2+0.5], [alpha1]*2, color = cm.gray(col), linestyle='--', linewidth=2.0)
    axes[1].plot([index1-0.5, index2+0.5], [alpha2]*2, color = cm.gray(col), linestyle='--', linewidth=2.0)

for bind, taus_tuple in zip(bands_indexes, mean_taus1, mean_taus2):
    tau1, tau2 = taus_tuple
    index1 = xvalues[sdata.relevant_peaks.index(band[0])]
    index2 = xvalues[sdata.relevant_peaks.index(band[1])]
    axes[2].plot([index1-0.5, index2+0.5], [tau1]*2, color = cm.gray(col),
                 linestyle='--', linewidth=2.0)
    axes[3].plot([index1-0.5, index2+0.5], [tau2]*2, color = cm.gray(col),
                 linestyle='--', linewidth=2.0)
    if tau2 == 0:
        continue

# plt.savefig( './Figure4/F4a.svg' )
