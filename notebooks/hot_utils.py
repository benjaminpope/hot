import numpy as np
import matplotlib.pyplot as plt
import glob
from oxksc.cbvc import cbv
import fitsio 

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