#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pycbc.psd
from pycbc import frame
from astropy.time import Time
import h5py
from pycbc.filter import sigma,sigmasq
from pycbc.waveform.generator import FDomainCBCGenerator,FDomainDetFrameGenerator
from tqdm import tqdm
import healpy as hp

import matplotlib as mpl
import matplotlib.pyplot as plt

# PLOTTING OPTIONS
fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = { 'axes.labelsize': 24,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 24,
          'legend.fontsize': 20,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.grid' : True,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14,
          'figure.figsize': fig_size}

mpl.rcParams.update(params)


# In[2]:


def radec2pix(ra, dec,NSIDE=32):
    return hp.pixelfunc.ang2pix(NSIDE, np.pi/2-dec, ra)#np.pi*2 - ra

def pix2radec(index,NSIDE=32):
    theta,phi=hp.pixelfunc.pix2ang(NSIDE, index)
    return phi, np.pi/2 - theta

class postprocess():
    def __init__(self,
                 gw_pe_path,
                 obrun):
        '''Function to postprocess GW runs, in particular to compute the exclusion distance
        
        Arguments:
        ------------
        gw_pe_path: str
            Path for posterior file
        obrun: str
            'o1' or 'o2' or 'o3a' or 'o3b'
        '''
        #Gravitational wave parameter estimation posterior file
        self.gw_pe = h5py.File(gw_pe_path,'r') 
        self.obrun = obrun
        self.det = list(self.gw_pe.attrs['analyzed_detectors'])
        self.trigger_time = self.gw_pe.attrs['trigger_time']
        #GW waveform generator
        self.generator = FDomainDetFrameGenerator(FDomainCBCGenerator, 
                                    self.trigger_time, 
                                    variable_args=['mass1', 'mass2', 'spin1z', 'spin2z', 
                                                   'ra', 'dec', 'inclination',
                                                   'distance','tc', 'coa_phase','polarization'], 
                                    detectors=self.det, 
                                    delta_f=1/4, f_lower=20., approximant='TaylorF2')
        self.gw_frame_type={'o1_H1':["H1_LOSC_16_V1","H1:GWOSC-16KHZ_R1_STRAIN"],
                           'o1_L1':["L1_LOSC_16_V1","L1:GWOSC-16KHZ_R1_STRAIN"],
                           'o2_H1':["H1_GWOSC_O2_16KHZ_R1","H1:GWOSC-16KHZ_R1_STRAIN"],
                           'o2_L1':["L1_GWOSC_O2_16KHZ_R1","L1:GWOSC-16KHZ_R1_STRAIN"],
                           'o2_V1':["V1_GWOSC_O2_16KHZ_R1","V1:GWOSC-16KHZ_R1_STRAIN"],
                           'o3a_H1':["H1_GWOSC_O3a_4KHZ_R1","H1:GWOSC-4KHZ_R1_STRAIN"],
                           'o3a_L1':["L1_GWOSC_O3a_4KHZ_R1","L1:GWOSC-4KHZ_R1_STRAIN"],
                           'o3a_V1':["V1_GWOSC_O3a_4KHZ_R1","V1:GWOSC-4KHZ_R1_STRAIN"],
                           'o3b_H1':["H1_GWOSC_O3b_4KHZ_R1","H1:GWOSC-4KHZ_R1_STRAIN"],
                           'o3b_L1':["L1_GWOSC_O3b_4KHZ_R1","L1:GWOSC-4KHZ_R1_STRAIN"],
                           'o3b_V1':["V1_GWOSC_O3b_4KHZ_R1","V1:GWOSC-4KHZ_R1_STRAIN"]}
        self.psd = self.make_psd()
    def make_psd(self):
        '''Power spectral density estimation for LIGO/Virgo
            
        Return:
        ------------
        psd: dict
            Dictionary for PSD for each inteferometer
            
        '''
        psd = {}
        obrun = self.obrun
        
        for d in self.det:
            psd_start = self.gw_pe.attrs[d+'_psd_segment'][0]
            psd_end = self.gw_pe.attrs[d+'_psd_segment'][1]
            frame_type = self.gw_frame_type[obrun+'_'+d][0]
            frame_channel = self.gw_frame_type[obrun+'_'+d][1]
            
            tseries = frame.query_and_read_frame(frame_type, frame_channel, psd_start, psd_end)
            seg_len = int(4 / tseries.delta_t)
            seg_stride = int(seg_len / 2)
            estimated_psd = pycbc.psd.welch(tseries, seg_len=seg_len, seg_stride=seg_stride)
            psd[d] = estimated_psd
        return psd
    
    def optimal_snr(self,ra,dec,pol):
        '''Optimal SNR for a specific set of parameters
        
        Arguments:
        ------------
        ra,dec,pol: float
            right ascension, declination and polarization angle of GW
        
        Return:
        ------------
        np.sqrt(snrsq): float
            Optimal SNR
            
        '''
        params = {'mass1':1.4,
         'mass2':1.4,
         'spin1z':0,
         'spin2z':0,
         'inlication':np.pi/6,
         'distance':100,
         'tc':self.trigger_time,
         'coa_phase':0}
        params.update({'ra':ra,
                      'dec':dec,
                      'polarization':pol})
        wf = self.generator.generate(**params)
        psd = self.psd
        snrsq = 0
        for d in self.det:
            snrsq += sigmasq(wf[d],psd[d],low_frequency_cutoff=20,high_frequency_cutoff=2048)
        return np.sqrt(snrsq)


# In[3]:


grb = h5py.File('grbdata/longgrb-skymap.hdf','r')

gwname_list = []
grbname_list = []
snr_list = []
for o in ['o1','o2','o3a','o3b']:
    search_result = pd.read_csv('search_result_'+o+'.csv')
    for i in range(len(search_result.index)):
        gwname = search_result['gwname'][i]
        grbname = search_result['grbname'][i]
        gwname_list.append(gwname)
        grbname_list.append(grbname)
        print(gwname,grbname)
        result = postprocess('/work/yifan.wang/grb/gwrun/'+o+'/result/'+gwname+'.hdf',o)
        healpix = grb[grbname][:]
        healpix[healpix<0] = 0
        snr = 0
        for i,v in tqdm(enumerate(healpix)):
            if v==0:
                continue
            ra,dec = pix2radec(i)
            for pol in np.linspace(0,np.pi*2,1000):
                snr += result.optimal_snr(ra,dec,pol)/1000*v
        snr_list.append(snr)

result_dict = {
            'gwname': gwname_list,
            'grbname': grbname_list,
            'snr': snr_list,
	    'ex_distance':np.array(snr_list)/8*100
          }
df= pd.DataFrame(result_dict)
df.to_csv('exclusion_distance.csv',index=False)
