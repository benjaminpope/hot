import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table, join, Column
from astropy.stats import LombScargle
import astropy.units as u                          # We'll need this later.
import warnings
from astropy.io import ascii
import glob, re, copy

import lightkurve
from lightkurve import KeplerLightCurveFile, KeplerLightCurve
from hot_utils import *

import matplotlib as mpl
mpl.style.use('seaborn-colorblind')

#To make sure we have always the same matplotlib settings
#(the ones in comments are the ipython notebook settings)

mpl.rcParams['figure.figsize']=(12.0,9.0)    #(6.0,4.0)
mpl.rcParams['font.size']=20               #10 
mpl.rcParams['savefig.dpi']= 200             #72 
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
from matplotlib import rc

colours = mpl.rcParams['axes.prop_cycle'].by_key()['color']

from argparse import ArgumentParser

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('kic', type=int)
    ap.add_argument('--data-dir', default='.', type=str)
    ap.add_argument('--save-dir', default='.', type=str)
    ap.add_argument('--plot-dir', default='.', type=str)
    ap.add_argument('--do-plots', action='store_true', default=True)
    ap.add_argument('--plot-format', type=str, default='png', choices=['pdf', 'png'], help='File format for plots')
    ap.add_argument('--planet-p-min', type=float, default=None)
    ap.add_argument('--planet-p-max', type=float, default=None)
    ap.add_argument('--outdir', type=str, default='plots/')


    args = ap.parse_args()

    do_all(args.kic,planet_p_range=(args.planet_p_min,args.planet_p_max),renormalize=True,outdir=args.outdir)