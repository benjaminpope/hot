import numpy as np
import matplotlib.pyplot as plt
import glob
from oxksc.cbvc import cbv
import fitsio 
from astropy.stats import LombScargle
import astropy.units as u                          # We'll need this later.
import copy

def find_cbv(quarter):
    fname = glob.glob('../data/kplr_cbv/*q%02d*-d25_lcbv.fits' % quarter)[0]
    return fname
    
def get_num(quarter):
    '''Get the file number for each quarter'''
    if quarter == 0:
        return 2009131105131
    elif quarter == 1:
        return 2009166043257
    elif quarter == 2:
        return 2009259160929
    elif quarter == 3:
        return 2009350155506
    elif quarter == 4: # two sets of collateral - this is released 2014-12-17 16:15:42
        return 2010078095331
    # elif quarter == 4: # two sets of collateral - this is released 2012-04-25 13:41:56
    #   return 2010078170814
    elif quarter == 5:
        return 2010174085026
    elif quarter == 6:
        return 2010265121752
    elif quarter == 7:
        return 2010355172524
    elif quarter == 8:
        return 2011073133259
    elif quarter == 9:
        return 2011177032512
    elif quarter == 10:
        return 2011271113734
    elif quarter == 11:
        return 2012004120508
    elif quarter == 12:
        return 2012088054726
    elif quarter == 13:
        return 2012179063303
    elif quarter == 14:
        return 2012277125453
    elif quarter == 15:
        return 2013011073258
    elif quarter == 16:
        return 2013098041711
    elif quarter == 17:
        return 2013131215648

def get_mod_out(channel):
    tab = Table.read('mod_out.csv')
    index = np.where(tab['Channel']==channel)
    mod, out = tab['Mod'][index], tab['Out'][index]
    return int(mod), int(out)


def match_cadences(cbvcads,lccads):
    indices =np.array([1 if j in lccads else 0 for j in cbvcads])
    return np.where(indices==1)[0]

def correct_quarter(lc,quarter):
    fname = find_cbv(quarter)
    cbvfile = fitsio.FITS(fname)
    m = (lc.quarter== quarter)

    cads = match_cadences(cbvfile[lc.channel]['CADENCENO'][:],lc[m].cadenceno)
    basis = np.zeros((lc[m].flux.shape[0],16))

    for j in range(16):
        basis[:,j] = cbvfile[lc.channel]['VECTOR_%d'% (j+1)][cads]
        
    corrected_flux, weights = cbv.fixed_nb(lc[m].flux, basis,doPlot = False)

    return corrected_flux

def sine_renormalize(lc,min_period=4./24.,max_period=30.):
    powers = []

    # get an overall mean model
    bestfreq, power = get_best_freq(lc,min_period=0.5,max_period=10)
    ytest_mean = LombScargle(lc.time, lc.flux-1, lc.flux_err).model(lc.time, bestfreq)
    current = (np.max(ytest_mean)-np.min(ytest_mean))/2.

    dummy = copy.copy(lc)

    for j, q in enumerate(np.unique(lc.quarter)):
        m = (lc.quarter == q)

        ytest = LombScargle(lc.time[m], lc.flux[m]-1., lc.flux_err[m]).model(lc.time[m], bestfreq)
        power = (np.max(ytest)-np.min(ytest))/2.#/np.median(dummy.flux[m])
        dummy.flux[m] = 1.+(dummy.flux[m]-1.)/(power/current)

        powers.append(power)
    
    return dummy, powers

def get_best_freq(lc,min_period=4./24., max_period=30.):
    lc2 = copy.copy(lc)

    frequency, power = LombScargle(lc2.time, lc2.flux, lc2.flux_err).autopower(minimum_frequency=1./max_period,
                                                                                         maximum_frequency=1./min_period, 
                                                                                         samples_per_peak=3)
    best_freq = frequency[np.argmax(power)]

    # refine
    for j in range(3):
        frequency, power = LombScargle(lc2.time, lc2.flux, lc2.flux_err).autopower(minimum_frequency=best_freq*(1-1e-3*(3-j)),
                                                                                         maximum_frequency=best_freq*(1+1e-3*(3-j)), 
                                                                                         samples_per_peak=1000)
        best_freq = frequency[np.argmax(power)]

    return best_freq, np.max(power)


def iterative_sine_fit(lc,nmax,min_period=4./24.*u.day, max_period=30.*u.day):
    ff, pp, noise = [], [], []
    y_fit = 0
    lc2 = copy.copy(lc)

    for j in range(nmax):
        best_freq, maxpower = get_best_freq(lc,min_period=min_period,max_period=max_period)
        
        ff.append(best_freq)
        pp.append(maxpower)
        y_fit += LombScargle(lc2.time, lc2.flux-1, lc2.flux_err).model(lc2.time, best_freq)
        lc2.flux = lc.flux - y_fit 
        noise.append(lc2.estimate_cdpp())
        
    return lc2, ff, pp, noise 