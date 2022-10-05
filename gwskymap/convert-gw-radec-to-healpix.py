#!/usr/bin/env python
# coding: utf-8

import h5py
import os
import numpy as np
from tqdm import tqdm
from pycbc.inference import io
import healpy as hp
from argparse import ArgumentParser

parser = ArgumentParser(description=__doc__)
parser.add_argument("--nth", type=int, required=True,
                    help="The nth partition segment to analyze.")
parser.add_argument("--N", type=int, required=True,
                    help="The total number of segments")
parser.add_argument("--obrun", type=str, required=True,
                    help="The observation run of LIGO/Virgo")
options = parser.parse_args()

NSIDE = 32


def pix2radec(index):
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, index)
    return phi, np.pi/2 - theta


def radec2pix(ra, dec):
    return hp.pixelfunc.ang2pix(NSIDE, np.pi/2-dec, ra)  # np.pi*2 - ra


def process(gwname, gwgps, ind, obrun):
    with h5py.File('/work/yifan.wang/grb/gwrun/gwskymap/' + obrun+'/'+
                       obrun+'_gwskymap'+'_'+str(ind)+'.hdf', 'w') as output:
        for nameidx, name in tqdm(enumerate(gwname)):
            directory = '/work/yifan.wang/grb/gwrun/'+obrun+'/result/'+name.decode()+'.hdf'
            fp = io.loadfile(directory, 'r')
            params = fp.read_samples(['ra', 'dec'])
            config = fp.read_config_file()
    
            # Simple Kernel Density Estimate
            W = np.ones(5)
            R = .02
    
            n = hp.nside2npix(NSIDE)
            m = np.zeros(n)
    
            for ra, dec in zip(params['ra'], params['dec']):
                vec = hp.pix2vec(NSIDE, radec2pix(ra, dec))
                for j, w in enumerate(W):
                    idx = hp.query_disc(NSIDE, vec, R*(j+1))
                    m[idx] += w
            m /= m.sum()
            m = m.astype(np.float32)
            output[name] = m
            output[name].attrs['gps'] = gwgps[nameidx]
            output[name].attrs['stat'] = float(config.get('trigger', 'stat'))


def low_high_index(nth, N, l):
    '''
    nth: the n-th segment
    N: total number of segments
    l: the length of array
    '''
    partitionl = l//N
    lown = nth*partitionl
    highn = l if nth == N-1 else (nth+1)*partitionl
    return lown, highn, highn-lown


nth = options.nth
obrun = options.obrun
gwname = []
gwgps = []
N = options.N
with h5py.File('/work/yifan.wang/grb/git-notebooks/gw_candidate_prod.hdf', 'r') as gwfile:
    for i in range(N):
        lowi, highi, _ = low_high_index(i, N, len(gwfile[obrun]['name']))
        gwname.append(gwfile[obrun]['name'][lowi:highi])
        gwgps.append(gwfile[obrun]['gps'][lowi:highi])

process(gwname[nth], gwgps[nth], nth, obrun)
