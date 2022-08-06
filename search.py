#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
from astropy.time import Time
from argparse import ArgumentParser
parser = ArgumentParser(description=__doc__)
parser.add_argument("--obrun", type=str, required=True,
                    help="The LIGO/Virgo Observation Run")

# PLOTTING OPTIONS
fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {'axes.labelsize': 24,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 24,
          'legend.fontsize': 20,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.grid': True,
          'text.usetex': True,
          'savefig.dpi': 100,
          'lines.markersize': 14,
          'figure.figsize': fig_size}

mpl.rcParams.update(params)

# Get times of GRBs
options = parser.parse_args()
obrun = options.obrun
grb_file = h5py.File('./grbdata/longgrb-skymap.hdf', 'r')

t_grb = []
name_grb = []
for k in grb_file.keys():
    t_grb.append(grb_file[k].attrs['gps'])
    name_grb.append(k)

t_grb = np.array(t_grb)
name_grb = np.array(name_grb)

days = (t_grb.max() - t_grb.min()) / 86400.0
npday_grb = len(t_grb) / days
swait_grb = 86400 / npday_grb
print('Total GRB span time:', days, 'days; average waiting time (s):', swait_grb)
print('There are', len(t_grb), 'GRBs')


gw_file = h5py.File(
    './gwdata/gwskymap_'+str(obrun)+'.hdf', 'r')

t_gw = []
name_gw = []
print('extract gw names...')
for k in tqdm(gw_file):
    t_gw.append(gw_file[k].attrs['gps'])
    name_gw.append(k)

t_gw = np.array(t_gw)
name_gw = np.array(name_gw)


arggw = np.argsort(t_gw)
arggrb = np.argsort(t_grb)

sort_t_gw = t_gw[arggw]
sort_name_gw = name_gw[arggw]

sort_t_grb = t_grb[arggrb]
sort_name_grb = name_grb[arggrb]

def coincs(gw, grb, window=10):
    '''Function to search for temporal coincident GW and GRB signals

    Arguments:
    --------
    gw: np.array
        Gravitational wave candidates GPS time (sorted)
    grb: np.array
        Gamma-Ray burst candidates GPS time (sorted)
    windown: int, unit:s (default: 10)
        Time delay between GW and GRB

    Return:
    --------
    gwfound: np.array
        GW index for an associated pair
    grbfound: np.array
        GRB index for an associated pair
    '''
    gw = gw.copy()
    grb = grb.copy()
    left = np.searchsorted(grb, gw)
    right = np.searchsorted(grb, gw + window)

    gwfound = np.where((right - left) > 0)[0]
    grbfound = left[gwfound]
    return gwfound, grbfound

def getbstat(gw_name, gw_file, grb_name, grb_file):
    '''Function to get ranking statistic for a temporal associated pair

    Arguments:
    ----------
    gw_name: str
        Gravitational wave candidate name
    gw_file: hdf5 file
        A hdf5 file storing the GW skymap
    grb_name: str
        Long Gamma-ray Burst candidate name
    grb_file: hdf5 file
        A hdf5 file storing the GRB skymap
    '''
    gw_map = gw_file[gw_name][:]
    gw_stat = gw_file[gw_name].attrs['stat']
    grb_map = grb_file[grb_name][:]

    gw_map = np.array(gw_map)
    grb_map = np.array(grb_map)

    B = np.sum(gw_map*grb_map)*len(gw_map)
    if B <= 0:
        return -np.inf
    return gw_stat + np.log(B)


# # Search for foreground candidates
print('start search ...')

ind_gw, ind_grb = coincs(sort_t_gw, sort_t_grb, 10)
cand_gwname = sort_name_gw[ind_gw]
cand_grbname = sort_name_grb[ind_grb]

ranking = []
for v_gwname, v_grbname in zip(cand_gwname, cand_grbname):
    bstat = getbstat(v_gwname, gw_file, v_grbname, grb_file)
    ranking.append(bstat)


# # Determine sliding time for background estimation
# expnum = 1
# window = expnum * swait_grb / len(t_gw) / 2.0
# print(window)

window = 200
tslide = np.arange(-5000, -5) * window
tslide2 = np.arange(5, 5000) * window
tslide = np.concatenate([tslide, tslide2])

print('tslide',tslide)

# Calculate the background
bg_gwind = []
bg_grbind = []

for slide in tqdm(tslide):
    gwi, grbi = coincs(sort_t_gw, sort_t_grb + slide)
    bg_gwind.append(gwi)
    bg_grbind.append(grbi)
bg_gwind = np.concatenate(bg_gwind)
bg_grbind = np.concatenate(bg_grbind)
print('Number of background:',len(bg_gwind))

bg_ranking = []

bg_gwname = sort_name_gw[bg_gwind]
bg_grbname = sort_name_grb[bg_grbind]

for v_gwname, v_grbname in tqdm(zip(bg_gwname, bg_grbname)):
    bstat = getbstat(v_gwname, gw_file, v_grbname, grb_file)
    bg_ranking.append(bstat)


#False alarm rate
bg_ranking = np.array(bg_ranking)
far = []
for i in ranking:
    far.append(len(bg_ranking[bg_ranking >= i])/len(tslide))

gw_gps=[]
gw_stat=[]
for name in cand_gwname:
    gw_stat.append(gw_file[name].attrs['stat'])
    gw_gps.append(gw_file[name].attrs['gps'])
grb_gps=[]
for name in cand_grbname:
    grb_gps.append(grb_file[name].attrs['gps'])


result={'gwname': cand_gwname,
          'gwtime': Time(gw_gps, format='gps', scale='utc').datetime64,
          'gwstat': gw_stat,
         'grbname': cand_grbname,
         'grbtime': Time(grb_gps, format='gps', scale='utc').datetime64,
          'rank_stat': ranking,
         'false_alarm_rate': far}
df=pd.DataFrame(result)
df.sort_values(by='false_alarm_rate', ascending=True, inplace=True)
df.to_csv('search_result_'+obrun+'.csv', index=False)

fig=plt.figure()
ax=fig.add_subplot(111)

low=-30
bins=np.linspace(low, np.max(bg_ranking)+1, 200)
counts, _=np.histogram(bg_ranking, bins=bins)

ax.hist(bins[:-1], bins, weights=counts/len(tslide),
        histtype="step", label='background')
ax.hist(ranking, bins=bins, alpha=0.7, color='grey', label='candidates')

ax.set_yscale('log')
ax.legend()
ax.set_xlim(-10,)
ax.set_xlabel('$\lambda_\mathrm{gw+grb}$')
ax.set_ylabel('Number of events / Obs')
fig.savefig(obrun+'.png', bbox_inches='tight')
