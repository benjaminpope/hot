import numpy as np
import matplotlib.pyplot as plt
import glob
from oxksc.cbvc import cbv
import fitsio 
from astropy.stats import LombScargle
import astropy.units as u                          # We'll need this later.
import copy
from scipy.ndimage import gaussian_filter1d as gaussfilt
from pybls import BLS
from matplotlib import rc


# planet search stuff

from k2ps.psearch import TransitSearch
from scipy.ndimage import binary_dilation
from pytransit import MandelAgol as MA
from pytransit import Gimenez as GM
from acor import acor

from exotk.utils.orbits import as_from_rhop, i_from_baew
from exotk.utils.likelihood import ll_normal_es
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplots, subplot


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

def stitch_lc_list(lcs,flux_type='PDCSAP_FLUX'):
    quarters = np.array([])
    channels = np.array([])
    for j, lci in enumerate(lcs):
        lci = lci.get_lightcurve(flux_type).remove_nans()
        lci = lci[lci.quality==0]
        lcs[j] = lci.normalize()
        quarters = np.append(quarters,lci.quarter*np.ones_like(lci.flux))
        channels = np.append(channels,lci.channel*np.ones_like(lci.flux))

    lc = copy.copy(lcs[0])
    for lci in lcs[1:]:
        lc = lc.append(copy.copy(lci))
        
    lc.quarter = quarters.astype('int')
    lc.channel = channels.astype('int')

    return lc

def match_cadences(cbvcads,lccads):
    indices =np.array([1 if j in lccads else 0 for j in cbvcads])
    return np.where(indices==1)[0]

def correct_quarter(lc,quarter):
    fname = find_cbv(quarter)
    cbvfile = fitsio.FITS(fname)
    m = (lc.quarter== quarter)
    channel = lc.channel[0]

    cads = match_cadences(cbvfile[channel]['CADENCENO'][:],lc[m].cadenceno)
    basis = np.zeros((lc[m].flux.shape[0],16))

    for j in range(16):
        basis[:,j] = cbvfile[channel]['VECTOR_%d'% (j+1)][cads]
        
    corrected_flux, weights = cbv.fixed_nb(lc[m].flux, basis, doPlot = False)

    return corrected_flux

def correct_all(lc):
    lc2 = copy.copy(lc)
    lc2.trposi = np.zeros_like(lc2.flux)
    for qq in np.unique(lc2.quarter.astype('int')):

        m = lc2.quarter==qq
        corrflux = correct_quarter(lc2,qq)
        lc2.trposi[m] = lc.flux[m] - corrflux + np.nanmedian(corrflux)
    return lc2


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
                                                                                         samples_per_peak=3,normalization='psd')
    best_freq = frequency[np.argmax(power)]

    # refine
    for j in range(3):
        frequency, power = LombScargle(lc2.time, lc2.flux, lc2.flux_err).autopower(minimum_frequency=best_freq*(1-1e-3*(3-j)),
                                                                                         maximum_frequency=best_freq*(1+1e-3*(3-j)), 
                                                                                         samples_per_peak=1000,normalization='psd')
        best_freq = frequency[np.argmax(power)]

    return best_freq, np.max(power)


def iterative_sine_fit(lc,nmax,min_period=4./24., max_period=30.):
    ff, pp, noise = [], [], []
    y_fit = 0
    lc2 = copy.copy(lc)

    for j in range(nmax):
        best_freq, maxpower = get_best_freq(lc2,min_period=min_period,max_period=max_period)
        
        ff.append(best_freq)
        pp.append(maxpower)
        y_fit += LombScargle(lc2.time, lc2.flux-1, lc2.flux_err).model(lc2.time, best_freq)
        lc2.flux = lc.flux - y_fit 
        noise.append(lc2.estimate_cdpp())

    lc2.trtime = y_fit + np.nanmedian(lc.flux)
    lc2.flux = lc.flux
    try:
        lc2.corr_flux = lc2.corr_flux - lc2.trtime + np.nanmedian(lc2.trtime)
    except:
        lc2.corr_flux = lc2.flux - lc2.trtime + np.nanmedian(lc2.trtime)
        
    return lc2, np.array(ff), np.array(pp), np.array(noise) 

def renorm_sde(bls,niter=3,order=2,nsig=2.5):
    '''
    In Pope et al 2016 we noted that the BLS has a slope towards longer periods. 
    To fix this we used the full ensemble of stars to build a binne median SDE as a function of period.
    With our smaller numbers here we just aggressively sigma-clip and fit a quadratic, and it seems to do basically ok. 
    '''
    trend = gaussfilt(bls.sde,20)
    sde = copy.copy(bls.sde)
    outliers = np.zeros(len(sde))

    for j in range(niter):
        outliers = np.abs(sde-trend)>(nsig*np.std(sde-trend))
        sde[outliers] += trend[outliers]
        trend = np.poly1d(np.polyfit(bls.period,sde,order))(bls.period)

    return trend 

def estimate_cdpp(lc, flux, transit_duration=13, savgol_window=101,
                  savgol_polyorder=2, sigma_clip=5.):
    """
    Copied from lightkurve: https://github.com/KeplerGO/lightkurve/

    Estimate the CDPP noise metric using the Savitzky-Golay (SG) method.
    A common estimate of the noise in a lightcurve is the scatter that
    remains after all long term trends have been removed. This is the idea
    behind the Combined Differential Photometric Precision (CDPP) metric.
    The official Kepler Pipeline computes this metric using a wavelet-based
    algorithm to calculate the signal-to-noise of the specific waveform of
    transits of various durations. In this implementation, we use the
    simpler "sgCDPP proxy algorithm" discussed by Gilliland et al
    (2011ApJS..197....6G) and Van Cleve et al (2016PASP..128g5002V).
    The steps of this algorithm are:
        1. Remove low frequency signals using a Savitzky-Golay filter with
           window length `savgol_window` and polynomial order `savgol_polyorder`.
        2. Remove outliers by rejecting data points which are separated from
           the mean by `sigma_clip` times the standard deviation.
        3. Compute the standard deviation of a running mean with
           a configurable window length equal to `transit_duration`.
    We use a running mean (as opposed to block averaging) to strongly
    attenuate the signal above 1/transit_duration whilst retaining
    the original frequency sampling.  Block averaging would set the Nyquist
    limit to 1/transit_duration.
    Parameters
    ----------
    transit_duration : int, optional
        The transit duration in units of number of cadences. This is the
        length of the window used to compute the running mean. The default
        is 13, which corresponds to a 6.5 hour transit in data sampled at
        30-min cadence.
    savgol_window : int, optional
        Width of Savitsky-Golay filter in cadences (odd number).
        Default value 101 (2.0 days in Kepler Long Cadence mode).
    savgol_polyorder : int, optional
        Polynomial order of the Savitsky-Golay filter.
        The recommended value is 2.
    sigma_clip : float, optional
        The number of standard deviations to use for clipping outliers.
        The default is 5.
    Returns
    -------
    cdpp : float
        Savitzky-Golay CDPP noise metric in units parts-per-million (ppm).
    Notes
    -----
    This implementation is adapted from the Matlab version used by
    Jeff van Cleve but lacks the normalization factor used there:
    svn+ssh://murzim/repo/so/trunk/Develop/jvc/common/compute_SG_noise.m
    """
    lc2 = copy.copy(lc)
    lc2.flux = flux
    return lc2.estimate_cdpp()


str_to_dt = lambda s: [tuple(t.strip().split()) for t in s.split(',')]
dt_lcinfo    = str_to_dt('epic u8, flux_median f8, Kp f8, flux_std f8, lnlike_constant f8, type a8,'
                         'acor_raw f8, acor_corr f8, acor_trp f8, acor_trt f8')
dt_blsresult = str_to_dt('sde f8, bls_zero_epoch f8, bls_period f8, bls_duration f8, bls_depth f8,'
                         'bls_radius_ratio f8, ntr u4')

class BasicSearch(TransitSearch):
    
    def __init__(self, d, inject=False,**kwargs):
        ## Keyword arguments
        ## -----------------
        self.nbin = kwargs.get('nbin', 2000)
        self.qmin = kwargs.get('qmin', 0.001)
        self.qmax = kwargs.get('qmax', 0.115)
        self.nf   = kwargs.get('nfreq', 10000)
        self.exclude_regions = kwargs.get('exclude_regions', [])

        ## Read in the data
        ## ----------------
        m  = np.isfinite(d.flux) & np.isfinite(d.time) # no mflags
        m &= ~binary_dilation((d.quality & 2**20) != 0)
        
        for emin,emax in self.exclude_regions:
            m[(d.time > emin) & (d.time < emax)] = 0
            
        try:
            self.Kp = d.Kp
        except:
            self.Kp = 12

        self.tm = MA(supersampling=12, nthr=1) 
        self.em = MA(supersampling=10, nldc=0, nthr=1)

        self.epic   = d.targetid
        self.time   = d.time[m]
        self.flux   = (d.flux[m] 
                       - d.trtime[m] + np.nanmedian(d.trtime[m]) 
                       - d.trposi[m] + np.nanmedian(d.trposi[m]))
        self.mflux   = np.nanmedian(self.flux)
        self.flux   /= self.mflux
        self.flux_e  = d.flux_err[m] / abs(self.mflux)

        self.flux_r  = d.flux[m] / self.mflux
        self.trtime = d.trtime[m] / self.mflux
        self.trposi = d.trposi[m] / self.mflux

        ## Initialise BLS
        ## --------------
        self.period_range = kwargs.get('period_range', (1.,40.))
        if self.nbin > np.size(self.flux):
            self.nbin = int(np.size(self.flux)/3)

        self.bls =  BLS(self.time, self.flux, self.flux_e, period_range=self.period_range, 
                        nbin=self.nbin, qmin=self.qmin, qmax=self.qmax, nf=self.nf, pmean='running_median')

        def ndist(p=0.302):
            return 1.-2*abs(((self.bls.period-p)%p)/p-0.5)

        def cmask(s=0.05):
            return 1.-np.exp(-ndist()/s)

        self.bls.pmul = cmask()

        try:
            ar,ac,ap,at = acor(self.flux_r)[0], acor(self.flux)[0], acor(self.trposi)[0], acor(self.trtime)[0]
        except RuntimeError:
            ar,ac,ap,at = np.nan,np.nan,np.nan,np.nan
        self.lcinfo = np.array((self.epic, self.mflux, self.Kp, self.flux.std(), np.nan, np.nan, ar, ac, ap, at), dtype=dt_lcinfo)

        self._rbls = None
        self._rtrf = None
        self._rvar = None
        self._rtoe = None
        self._rpol = None
        self._recl = None

        ## Transit fit pv [k u t0 p a i]
        self._pv_bls = None
        self._pv_trf = None
        
        self.period = None
        self.zero_epoch = None
        self.duration = None
        
    def run_bls(self):
        b = self.bls
        r = self.bls()
        new_sde = b.sde-renorm_sde(b)
        bper = b.period[np.argmax(new_sde)]
        bsde = np.max(new_sde)

        b = BLS(self.time, self.flux, self.flux_e, period_range=(bper*0.98,bper*1.02), 
                        nbin=self.nbin, qmin=self.qmin, qmax=self.qmax, nf=self.nf, pmean='running_median')# iterate once to get a better initial guess fit

        r = b()
        new_sde = b.sde-renorm_sde(b)
        bper = b.period[np.argmax(new_sde)]
        bsde = np.max(new_sde)


        bper = b.period[np.argmax(b.sde-renorm_sde(b))]
        self._rbls = np.array((bsde, b.tc, bper, (b.t2-b.t1), r.depth, np.sqrt(r.depth),
                            np.floor(np.diff(self.time[[0,-1]])[0]/bper)), dt_blsresult)
        self._pv_bls = [bper, b.tc, np.sqrt(r.depth), as_from_rhop(2.5, bper), 0.1]
        self.create_transit_arrays()
        self.lcinfo['lnlike_constant'] = ll_normal_es(self.flux, np.ones_like(self.flux), self.flux_e)
        self.period = bper
        self.zero_epoch = b.tc
        self.duration = b.t2-b.t1

def plot_all(ts):
    PW,PH = 8.27, 11.69
    rc('axes', labelsize=7, titlesize=8)
    rc('font', size=6)
    rc('xtick', labelsize=7)
    rc('ytick', labelsize=7)
    rc('lines', linewidth=1)
    fig = plt.figure(figsize=(PW,PH))
    gs1 = GridSpec(3,3)
    gs1.update(top=0.98, bottom = 2/3.*1.03,hspace=0.07,left=0.07,right=0.96)
    gs = GridSpec(4,3)
    gs.update(top=2/3.*0.96,bottom=0.04,hspace=0.35,left=0.07,right=0.96)

    ax_lcpos = subplot(gs1[0,:])
    ax_lctime = subplot(gs1[1,:],sharex=ax_lcpos)
    ax_lcwhite = subplot(gs1[2,:],sharex=ax_lcpos)
    ax_lcfold = subplot(gs[2,1:])
    ax_lnlike = subplot(gs[1,2])
    ax_lcoe   = subplot(gs[0,1]),subplot(gs[0,2])
    ax_sde    = subplot(gs[3,1:])
    ax_transits = subplot(gs[1:,0])
    ax_info = subplot(gs[0,0])
    ax_ec = subplot(gs[1,1])
    axes = [ax_lctime,ax_lcpos,ax_lcwhite, ax_lcfold, ax_lnlike, ax_lcoe[0], ax_lcoe[1], ax_sde, ax_transits, ax_info, ax_ec]

    ts.plot_lc_pos(ax_lcpos)
    ts.plot_lc_time(ax_lctime)
    ts.plot_lc_white(ax_lcwhite)
    ts.plot_eclipse(ax_ec)
    ts.plot_lc(ax_lcfold)
    ts.plot_fit_and_eo(ax_lcoe)
    ts.plot_info(ax_info)
    ts.plot_lnlike(ax_lnlike)
    ts.plot_sde(ax_sde)
    ts.plot_transits(ax_transits)
    ax_lnlike.set_title('Ln likelihood per orbit')
    ax_transits.set_title('Individual transits')
    ax_ec.set_title('Secondary eclipse')
    ax_lcoe[0].set_title('Folded transit and model')
    ax_lcoe[1].set_title('Even and odd transits')
    ax_lcfold.set_title('Folded light curve')

