# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Data"]

import os
import h5py
import copy
import fitsio
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from tqdm import tqdm

from lightkurve import KeplerLightCurve
from _compute import compute_hypotheses

# from ..gp_heuristics import estimate_tau, kernel, optimize_gp_params


class ketuLightCurve(KeplerLightCurve):

    def prepare(self, basis, nbasis=150, sigma_clip=7.0, max_iter=10,
                tau_frac=0.25, lam=1.0):
        self.lam = lam

        # Normalize the data.
        self.flux = self.flux / np.median(self.flux) - 1.0
        self.flux *= 1e3  # Convert to ppt.

        # Estimate the uncertainties.
        self.ivar = 1.0 / np.median(np.diff(self.flux) ** 2)
        self.ferr = np.ones_like(self.flux) / np.sqrt(self.ivar)

        # Load the prediction basis.
        self.basis = basis

        # Build the initial kernel matrix.
        self.tau_frac = tau_frac
        print('Building kernels')
        self.build_kernels()
        print('Kernel built!')

        # Do a few rounds of sigma clipping.
        if sigma_clip > 0:
            print('Sigma_clipping')
            m1 = np.ones_like(self.flux, dtype=bool)
            m2 = np.zeros_like(self.flux, dtype=bool)
            nums = np.arange(len(self.flux))
            count = m1.sum()
            for i in tqdm(range(max_iter)):
                inds = (nums[m1, None], nums[None, m1])
                alpha = np.linalg.solve(self.K[inds], self.flux[m1])
                mu = np.dot(self.K_0[:, m1], alpha)

                # Mask the bad points.
                r = self.flux - mu
                std = np.sqrt(np.median(r ** 2))
                m1 = np.abs(r) < sigma_clip * std
                m2 = r > sigma_clip * std

                if m1.sum() == count:
                    break
                count = m1.sum()

            # Save the sigma clipping mask.
            self.sig_clip = m1[~m2]
            print('Sigma clipping complete')
        else:
            print('No sigma clipping')
            m2 = np.isfinite(self.flux)

        # Force contiguity.
        self.time = np.ascontiguousarray(self.time[m2], dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux[m2], dtype=np.float64)
        self.ferr = np.ascontiguousarray(self.ferr[m2], dtype=np.float64)
        self.basis = np.ascontiguousarray(self.basis[:, m2], dtype=np.float64)
        self.quality = np.ascontiguousarray(self.quality[m2], dtype=int)

        # # Find outliers.
        # print('Finding outliers')
        # for _ in range(2):
        #     self.build_kernels()
        #     mu = np.dot(self.K_0, np.linalg.solve(self.K, self.flux))
        #     delta = np.diff(self.flux - mu)
        #     absdel = np.abs(delta)
        #     mad = np.median(absdel)
        #     m = np.zeros(self.m.sum(), dtype=bool)
        #     m[1:-1] = absdel[1:] > sigma_clip * mad
        #     m[1:-1] &= absdel[:-1] > sigma_clip * mad
        #     m[1:-1] &= np.sign(delta[1:]) != np.sign(delta[:-1])

        #     # Remove the outliers and finalize the dataset.
        #     m = ~m
        #     self.m[self.m] = m
        #     self.sig_clip = self.sig_clip[m]
        #     self.time = np.ascontiguousarray(self.time[m], dtype=np.float64)
        #     self.flux = np.ascontiguousarray(self.flux[m], dtype=np.float64)
        #     self.ferr = np.ascontiguousarray(self.ferr[m], dtype=np.float64)
        #     self.quality = np.ascontiguousarray(self.quality[m], dtype=int)
        #     self.basis = np.ascontiguousarray(self.basis[:, m],
        #                                       dtype=np.float64)
        # print('Outliers masked')
        print('Rebuilding kernels')
        self.build_kernels(mask=np.isfinite(self.flux))

        # Precompute some factors.
        print('Cholesky decomposition')
        self.factor = cho_factor(self.K)
        self.alpha = cho_solve(self.factor, self.flux)
        print('Decomposed!')

        # Pre-compute the base likelihood.
        self.ll0 = self.lnlike()

    def build_kernels(self, mask=None, optimize=False):
        if mask is None:
            mask = np.ones(len(self.flux), dtype=bool)
        self.K_b = np.dot(self.basis.T, self.basis*self.lam)
        # if self.gp:
        #     tau = self.tau_frac * estimate_tau(self.time[mask],
        #                                        self.flux[mask])
        #     print("tau = {0}".format(tau))
        #     if optimize:
        #         K_b = np.dot(self.basis[:, mask].T,
        #                      self.basis[:, mask]*self.lam)
        #         amp, tau = optimize_gp_params(tau, K_b, self.time[mask],
        #                                       self.flux[mask], self.ferr[mask])
        #     else:
        #         amp = np.var(self.flux)
        #     self.K_t = amp * kernel(tau, self.time)
        #     self.K_0 = self.K_b + self.K_t
        # else:
        self.K_0 = self.K_b
        self.K = np.array(self.K_0)
        self.K[np.diag_indices_from(self.K)] += self.ferr**2

    def lnlike_eval(self, y):
        return -0.5 * np.dot(y, cho_solve(self.factor, y))

    def grad_lnlike_eval(self, y, dy):
        alpha = cho_solve(self.factor, y)
        return -0.5 * np.dot(y, alpha), np.dot(alpha, dy)

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.dot(self.flux, self.alpha)

        # Evaluate the transit model.
        m = model(self.time)
        if np.all(m == 0.0):  # m[0] != 0.0 or m[-1] != 0.0 or
            return 0.0, 0.0, 0.0

        Km = cho_solve(self.factor, m)
        Ky = self.alpha
        ivar = np.dot(m, Km)
        depth = np.dot(m, Ky) / ivar
        r = self.flux - m*depth
        ll = -0.5 * np.dot(r, Ky - depth * Km)
        return ll - self.ll0, depth, ivar

    def search_lnlike(self, model=None):
        return self.lnlike(model=model)

    def predict(self, y=None):
        if y is None:
            y = self.flux
        return np.dot(self.K_0, cho_solve(self.factor, y))

    def predict_t(self, y):
        return np.dot(self.K_t, cho_solve(self.factor, y))

    def predict_b(self, y):
        return np.dot(self.K_b, cho_solve(self.factor, y))


    def search_one_d(self,min_period=1.0,max_period=40.,dt=0.01,durations=[0.05],alpha=500*np.log(100),min_transits=3,time_spacing=(0.05)):
        
        tmin, tmax = np.nanmin(self.time), np.nanmax(self.time)
        try:
            dt = time_spacing
        except:
            dt = 0.5*np.nanmedian(np.abs(self.time-np.roll(self.time,1)))

        time_grid = np.arange(tmin, tmax, dt)

        dll_grid = np.zeros((len(time_grid), len(durations)))
        depth_grid = np.zeros_like(dll_grid)
        depth_ivar_grid = np.zeros_like(dll_grid)

        i = np.arange(len(time_grid))
        imn, imx = i.min(), i.max()

        compute_hypotheses(self.search_lnlike, time_grid[imn:imx], np.array(durations),
                           depth_grid[imn:imx], depth_ivar_grid[imn:imx],
                           dll_grid[imn:imx])

        self.tmin = tmin
        self.tmax = tmax
        self.time_spacing = dt
        self.depth_1d = depth_grid
        self.dll_1d = dll_grid
        self.depth_ivar_1d = depth_ivar_grid
        self.mean_time_1d = 0.5*(tmin+tmax)

        print('1D Search Complete')

    def search_2d(self,min_transits=3):
        dt = 0.5*np.min(self.durations)
        results = grid_search(min_transits, self.alpha,
                      self.tmin, self.tmax, self.time_spacing, self.depth_1d,
                      self.depth_ivar_1d, self.dll_1d, self.periods, dt)
        self.t0_2d, self.phic_same, self.phic_same_2, self.phic_variable, self.depth_2d, self.depth_ivar_2d \
            = results


