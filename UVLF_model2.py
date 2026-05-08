########################################################################
# UV Luminosity Function Likelihood
# Based on original likelihood from https://github.com/NNSSA/GALLUMI_public.
# Remodelled for COBAYA mcmc sampler
# Written for Cobaya sampler on Oct 5, 2025
# Authors: Sai Chaitanya Tadepalli and Siu Cheung Lam
# Affiliation: Indiana University, Bloomington, USA
# Distribution by Sai Chaitaya Tadepalli
# Please cite 2110.XXXX and 2110.XXXX when using this likelihood
########################################################################

########################################################################
# Features: 
# Combined HST + JWST support
# Sheth-Tormen and Sharp-k window functions
# Speed-ups by vectorization/caching
# Asymmetric-error likelihood (sigma_up, sigma_lo).
# Added z-dependence for alphastar. Can be extended to other UVLF params
##########################################################################

from cobaya.likelihood import Likelihood

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.special import erf
import warnings
from pathlib import Path

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

try:
    from scipy.integrate import simpson as simps  
except ImportError:
    from scipy.integrate import simps as simps
    warnings.warn('Using "simps" instead of "simpson". Upgrade scipy for newer features')


class UVLF(Likelihood):
    """
    Combined HST + JWST UVLF likelihood.

    Expected data file formats (space- or tab-delimited; comments allowed):
        Columns: z, MUV, dM, phi, sigma_up, sigma_lo
    Paths used below can be adjusted via `base`.
    """
    # ---- options configurable from YAML ----
    CV: float = 0.2        # cosmic variance floor for the UVLF data
    z_max: float = 12.5   # max z data to be used for JWST
    window: str = 'tophat'    # default window function (Set 'sharpk' to use sharp-k window function)
    Anorm: float = None   # defaults to 1 for tophat window
    qnorm: float = None   # defaults to 1 for tophat window
    cnorm: float = None   # defaults to 1 for tophat window
    # ---------- Initialization ----------
    def initialize(self):
        if str(self.window).lower() == "sharpk":
            if self.Anorm is None or self.qnorm is None or self.cnorm is None:
                raise ValueError(
                    "For window='sharpk', Anorm, qnorm, and cnorm must be set explicitly in the YAML."
                )

        # If top-hat, use defaults if omitted
        if self.Anorm is None:
            self.Anorm = 1.0
        if self.qnorm is None:
            self.qnorm = 1.0
        if self.cnorm is None:
            self.cnorm = 1.0
        # ---- Integration grids ----
        self.points, self.weights = np.polynomial.legendre.leggauss(25)  # Gaussian quad for r(z)

        # Power spectrum k-grid (log spaced)
        self.k_max = 40.0
        self.kSpace = np.exp(np.linspace(np.log(1e-3), np.log(self.k_max), 1000))
        
        # log-k weights for ∫ dlnk ...
        self.lnk = np.log(self.kSpace)
        self.dlnk = np.diff(self.lnk, prepend=self.lnk[0])
        self.wlnk = self._trapz_weights(self.lnk)  # trapezoid weights in ln k
        self._w2_key = None
        self._W2_cached = None     # shape (Nk, NM)
        self._R_base_cached = None # shape (NM,)

        # ---- n_eff probe points (preselect nearest grid indices) ----
        # self._i1 = int(np.argmin(np.abs(self.kSpace - 1.0)))    # 1/Mpc
        # self._i2 = int(np.argmin(np.abs(self.kSpace - 15.0)))   # 1/Mpc
        # self._lnk_ratio = np.log(self.kSpace[self._i2] / self.kSpace[self._i1])        

        # Halo-mass grid (log spaced)
        self.Mhalos = np.geomspace(1e8, 1e14, 1000)   
        self.wM = self._trapz_weights(self.Mhalos)   # trapezoid weights over M

        # set path
        BASE = Path(__file__).resolve().parent
        # ---- Load UVLF data ----
        
        # HST data format (columns: z, MUV, dM, phi, sigma_up, sigma_lo)
        self.UVLF_HST = np.loadtxt(BASE / "UVLF_HST.txt")
        minError = self.CV  # 20% Cosmic Variance floor
        self.UVLF_HST[:, 4] = np.maximum(minError * self.UVLF_HST[:, 3], self.UVLF_HST[:, 4])
        self.UVLF_HST[:, 5] = np.maximum(minError * self.UVLF_HST[:, 3], self.UVLF_HST[:, 5])
        self.zs_hst = np.unique(self.UVLF_HST[:, 0])

        # JWST 
        try:
            self.UVLF_JWST = np.loadtxt(BASE / "UVLF_JWST.txt")
            self.UVLF_JWST = self.UVLF_JWST[self.UVLF_JWST[:, 0] <= self.z_max]
            self.zs_jwst = np.unique(self.UVLF_JWST[:, 0])    
        except OSError:
            self.UVLF_JWST = np.empty((0, 6))
            self.zs_jwst = np.array([])

        # ---- Dust beta(z) functions ----
        betadata = np.loadtxt(BASE / "Beta_parameters.txt", unpack=True)
        self.betainterp = PchipInterpolator(betadata[0], betadata[1])
        self.dbetadMUVinterp = PchipInterpolator(betadata[0], betadata[2])

        # ---- Apply dust / bin-width corrections to HST ----
        if self.UVLF_HST.size:
            dust_corr, bin_corr, LF_corr = [], [], []
            for item in self.UVLF_HST:
                z, MUV, bin_width = item[0], item[1], item[2]
                new_bin_width = bin_width - self.AUV(z, MUV + bin_width / 2) + self.AUV(z, MUV - bin_width / 2)
                dust_corr.append(self.AUV(z, MUV))
                bin_corr.append(new_bin_width)
                LF_corr.append(bin_width / new_bin_width)

            self.UVLF_HST[:, 1] -= np.array(dust_corr)
            self.UVLF_HST[:, 2] = np.array(bin_corr)
            self.UVLF_HST[:, 3] *= np.array(LF_corr)
            self.UVLF_HST[:, 4] *= np.array(LF_corr)
            self.UVLF_HST[:, 5] *= np.array(LF_corr)

        # ---- Parse constants ----
        self.parse_assign(BASE / "UVLF_ST_model2.data", 'UVLF_HST_ST_model2')
        
        # JWST reference cosmology (Donnan+ 2024)
        self.Omega_m_JWST = 0.3
        self.h_JWST = 0.7  # H0=70 => h=0.7

        # JWST redshift half-bin widths for AP
        self._jwst_dz_half = {9.0: 0.5, 10.0: 0.5, 11.0: 0.5, 12.5: 1.0, 14.5: 1.0}   # add here, to extend for higher 'z'

        # Union of redshifts
        self.zs_all = np.sort(np.unique(np.concatenate([self.zs_hst, self.zs_jwst])))

        # Small cache for comoving distances
        self._rcomov_cache = {}

        return

    # ---------- Utilities ----------
    @staticmethod
    def _trapz_weights(x):
        w = np.empty_like(x)
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
        w[0] = 0.5 * (x[1] - x[0])
        w[-1] = 0.5 * (x[-1] - x[-2])
        return w

    # Beta function for the dust extinction (HST)
    def betaAverage(self, z, MUV):
        if MUV < -19.5:
            return self.dbetadMUVinterp(z) * (MUV + 19.5) + self.betainterp(z)
        return (self.betainterp(z) + 2.33) * np.exp(
            (self.dbetadMUVinterp(z) * (MUV + 19.5)) / (self.betainterp(z) + 2.33)
        ) - 2.33

    # Dust extinction parameter (only applied where measured)
    def AUV(self, z, MUV):
        # eq. 3.3 in https://journals.aps.org/prd/pdf/10.1103/PhysRevD.105.043518
        # dust extinction neglected for z>8
        if z < 2.5 or z > 8:
            return 0.0
        sigmabeta = 0.34
        return max(
            0.0,
            4.54
            + 0.2 * np.log(10) * (2.07 ** 2) * (sigmabeta ** 2)
            + 2.07 * self.betaAverage(z, MUV),
        )

    # Comoving radial distance with Gaussian-Legendre sub integration (cached)
    def rcomoving(self, z, Omega_m, h):
        key = (float(z), float(Omega_m), float(h))
        r = self._rcomov_cache.get(key)
        if r is not None:
            return r

        def Ez_inv(x):
            return 1.0 / np.sqrt(Omega_m * (1.0 + x) ** 3 + 1.0 - Omega_m)

        sub = (z - 0.0) / 2.0
        add = (z + 0.0) / 2.0
        val = 0.0 if sub == 0.0 else sub * np.dot(Ez_inv(sub * self.points + add), self.weights)
        r = self.c * val / (100.0 * h)
        self._rcomov_cache[key] = r
        return r

    # ---------- Alcock-Paczynski corrections ----------
    def _AP_effect_generic(self, UV_table, zs_unique, Omega_m_ref, h_ref, dz_half_getter):
        if UV_table.size == 0:
            return UV_table

        Omega_m = self.provider.get_param("Omega_m")
        hubble  = self.provider.get_param("h")

        UV = UV_table.copy()
        for z in zs_unique:
            dz = float(dz_half_getter(float(z)))

            r_ref_hi = self.rcomoving(z + dz, Omega_m_ref, h_ref)
            r_ref_lo = self.rcomoving(z - dz, Omega_m_ref, h_ref)
            r_mod_hi = self.rcomoving(z + dz, Omega_m, hubble)
            r_mod_lo = self.rcomoving(z - dz, Omega_m, hubble)

            Vratio = (r_ref_hi**3 - r_ref_lo**3) / (r_mod_hi**3 - r_mod_lo**3)

            sel = (UV[:, 0] == z)
            UV[sel, 3:] *= Vratio

            r_ratio = self.rcomoving(z, Omega_m, hubble) / self.rcomoving(z, Omega_m_ref, h_ref)
            UV[sel, 1] -= 5.0 * np.log10(r_ratio)

        return UV

    def AP_effect_HST(self):
        return self._AP_effect_generic(
            self.UVLF_HST, self.zs_hst,
            self.Omega_m_HST, self.h_HST,
            dz_half_getter=lambda z: 0.5
        )

    def AP_effect_JWST(self):
        return self._AP_effect_generic(
            self.UVLF_JWST, self.zs_jwst,
            self.Omega_m_JWST, self.h_JWST,
            dz_half_getter=lambda z: self._jwst_dz_half.get(z, 0.5)
        )

    def AP_effect_all(self):
        UV_HST_corr = self.AP_effect_HST() if self.UVLF_HST.size else np.empty((0, 6))
        UV_JWST_corr = self.AP_effect_JWST() if self.UVLF_JWST.size else np.empty((0, 6))
        if UV_HST_corr.size or UV_JWST_corr.size:
            self.UVLF_data = np.vstack([UV_HST_corr, UV_JWST_corr])
        else:
            self.UVLF_data = np.empty((0, 6))
        return self.UVLF_data

    # ---------- Windows / variances / HMF ----------
    @staticmethod
    def wTophat(kR):
        # real-space tophat Fourier window
        kr = kR + 1e-30
        return 3.0 * (np.sin(kr) - kr * np.cos(kr)) / (kr ** 3)

    def _ensure_W2_cache(self, Omega_m, h):
        key = (float(Omega_m), float(h))
        if self._w2_key == key:
            return
        rhoM = (h**2) * Omega_m * self.rho_crit
        R_base = (3.0 * self.Mhalos / (4.0 * np.pi * rhoM))**(1.0/3.0)
        kR = np.outer(self.kSpace, R_base)           # (Nk, NM)
        W2 = self.wTophat(kR)**2                     # (Nk, NM)
        self._w2_key = key
        self._W2_cached = W2 
        self._R_base_cached = R_base 

    def _sigma_from_cached_W2(self, Pkz):
        # σ²(R) = ∫ dlnk Δ²(k,z) W²(kR)
        Delta2 = (self.kSpace**3) * Pkz / (2.0*np.pi**2)     # (Nk,)
        # (Nk,) -> weight by wlnk, then BLAS dot with cached W2: (Nk,) @ (Nk,NM) -> (NM,)
        sig2 = (self.wlnk * Delta2) @ self._W2_cached
        return np.sqrt(np.maximum(sig2, 0.0))

    def sigma_all(self, R_all, Pkz):
        """
        σ(R) with tophat: ∫ dlnk Δ²(k) W_th^2(kR).
        Use np.trapz over ln k (more accurate than left-Riemann dlnk sum).
        """
        k   = self.kSpace
        lnk = self.lnk
        Delta2 = (k**3) * Pkz / (2.0 * np.pi**2)     # (Nk,)
        kR = np.outer(k, R_all)                      # (Nk, NM)
        W2 = self.wTophat(kR)**2                     # (Nk, NM)
        sig2 = np.trapz(Delta2[:, None]*W2, x=lnk, axis=0)
        return np.sqrt(np.maximum(sig2, 0.0))

    def sigma_sharpk_all(self, R_all, Pkz, c_edge=1.0):
        k    = self.kSpace
        lnk  = self.lnk
        Delta2 = (k**3) * Pkz / (2.0 * np.pi**2)     # (Nk,)

        # cumulative integral in ln k using trapezoid
        I = np.empty_like(Delta2)
        I[0] = 0.0
        I[1:] = cumtrapz(Delta2, lnk)             # ∫ dlnk' Δ²

        # interpolate I(ln k) with a shape-preserving spline
        I_of_lnk = PchipInterpolator(lnk, I, extrapolate=False)

        lnkc = np.log(np.clip(c_edge/np.asarray(R_all), k[0], k[-1]))
        sig2 = I_of_lnk(lnkc)                        # already ∫^{kc} dlnk Δ²
        # For kc > k_max, I_of_lnk returns NaN (since extrapolate=False); clamp to total:
        sig2 = np.where(np.isfinite(sig2), sig2, I[-1])
        return np.sqrt(np.maximum(sig2, 0.0))
    
    def sharpk_sigma_and_D(self, R_all, Pkz, c_edge=1.0):
        """
        Sharp-k filter (k-space tophat):
            σ^2(R) = ∫^{k_c} dlnk Δ²(k),   k_c = c_edge / R
        Analytic derivative:
            D(M) ≡ -∂lnσ/∂lnM = Δ²(k_c) / [6 σ^2(R)]
        Returns:
            sigma : (NM,)  — rms fluctuation
            D     : (NM,)  — slope entering the mass function
        """
        k     = self.kSpace
        lnk   = self.lnk
        Delta2 = (k**3) * Pkz / (2.0*np.pi**2)              # (Nk,)
    
        # cumulative integral I(lnk) = ∫ dlnk' Δ²
        I = np.empty_like(Delta2)
        I[0] = 0.0
        I[1:] = cumtrapz(Delta2, lnk)
    
        # interpolants for σ^2 and Δ²
        F_I  = PchipInterpolator(lnk, I)                    # monotone, C^1
        F_D2 = PchipInterpolator(lnk, Delta2)
    
        # cutoff wavenumbers and evaluate
        kc    = np.clip(c_edge / np.asarray(R_all), k[0], k[-1])
        lnkc  = np.log(kc)
        sig2  = F_I(lnkc)
        sig2  = np.maximum(sig2, 0.0)
        sigma = np.sqrt(sig2)
    
        # analytic D(M) = Δ²(kc) / (6 σ^2)
        tiny  = 1e-300
        D     = F_D2(lnkc) / (6.0 * np.maximum(sig2, tiny))
    
        return sigma, D
    
    def HMF_all(self, z, Pk_interp):
        # cosmology pieces        
        h        = self.provider.get_param("h")
        Omega_m  = self.provider.get_param("Omega_m")
        rhoM     = (h ** 2) * Omega_m * self.rho_crit

        # ensure W2 (tophat) is ready ONCE per (Ωm,h)
        self._ensure_W2_cache(Omega_m, h)

        # P(k,z) once on the precomputed k-grid
        Pkz = Pk_interp.P(z=z, k=self.kSpace) #.astype(np.float64)
     
        lnM = np.log(self.Mhalos)

        # Sheth–Tormen multiplicity factors (keep your assignments of A,q,c)
        def f_ST(nu, A, a, p):
            return A * np.sqrt(2.0 * a / np.pi) * nu * np.exp(-0.5 * a * nu**2) * (1.0 + (a * nu**2) ** (-p))
        
        An = self.Anorm
        qn = self.qnorm
        cn = self.cnorm
        
        window = self.window
        # ---------------------------
        # Standard top-hat ST   # Tophat branch: (An, qn) = (1,1)
        # ---------------------------
        if window == "tophat": 
        
            sigma = self._sigma_from_cached_W2(Pkz)
    
            D = -np.gradient(np.log(sigma), lnM)
    
        # ---------------------------
        # Sharp-k
        # ---------------------------
        elif window == "sharpk":
    
            sigma, D = self.sharpk_sigma_and_D(
                self._R_base_cached,
                Pkz,
                c_edge=cn
            )
    
        nu = self.deltaST / sigma

        f = f_ST(
            nu,
            self.AST * float(An),
            self.aST * float(qn),
            self.pST
        )
    
        hmf = (rhoM / (self.Mhalos**2)) * f * D


        return hmf

    # ---------- M_UV mapping and integrated Gaussian ----------
    @staticmethod
    def MUV_from_Mh_vec(z, Mh, Hubble, alphastar_icept, betastar, epsilonstar_icept, Mc_icept
                        , alphastar_slope, epsilonstar_slope, Mc_slope
                        , kappaUV, invMpctoinvYear):
        
        epsilonstar = 10 ** (epsilonstar_slope * np.log10((1 + z) / (1 + 7)) + epsilonstar_icept)

        Mc = 10 ** ( Mc_slope *np.log10((1 + z) / (1 + 7)) + Mc_icept ) 

        alphastar = alphastar_slope * np.log10((1 + z) / (1 + 7)) + alphastar_icept

        MhMc = (Mh / Mc) 

        Mhdot = (
                (1.0)
                * Hubble
                * invMpctoinvYear
                * Mh
                )

        den = ((MhMc) ** alphastar + (MhMc) ** betastar) * kappaUV
        num = epsilonstar * Mhdot 
        return -2.5 * np.log10(num / den) + 51.63

    @staticmethod
    def first_integrand(MUV, width, MUV_av, sigma_MUV):
        # original symmetric bin integral (kept unchanged)
        return 0.5 * (erf((MUV_av - MUV + width / 2.0) / (sigma_MUV * np.sqrt(2.0))) -
                      erf((MUV_av - MUV - width / 2.0) / (sigma_MUV * np.sqrt(2.0))))

    # ----------------- Provider requirements ---------------
    def get_requirements(self):
        zs_req = self.zs_all if self.zs_all.size else np.array([6.0])
        return {
            "Pk_interpolator": {
                "z": zs_req,
                "k_max": self.k_max,
                "nonlinear": False,
                "vars_pairs": [("delta_tot", "delta_tot")]    # get total matter power spectrum
            },
            "omega_b": None,
            "h": None,
            "Hubble": {'z': zs_req},
            "Omega_m": None,
        }

    # ---------- Log-likelihood with asymmetric errors ----------
    def logp(self, _derived=None, **params):
        
        # --- separate scatters ---
        sigma_MUV_hst  = params['sigma_MUV_hst']   # z < 9
        sigma_MUV_jwst = params['sigma_MUV_jwst']  # z >= 9
        _ = self.provider.get_param('omega_b')  # keep as dependency if used upstream

        alphastar = params['alphastar_slope'] * np.log10((1 + self.zs_all) / (1 + 7)) + params['alphastar_icept']
            
        if np.any(alphastar > -0.01):   # check to ensure alphastar is always < 0 across all z-bins
            if _derived is not None:
                _derived["chi2_hst"] = float(-np.inf)
                _derived["chi2_jwst"] = float(-np.inf)

            return -np.inf

        data = self.AP_effect_all()
        if data.size == 0:
            return 0.0

        Pk_interp = self.provider.get_Pk_interpolator(var_pair=("delta_tot", "delta_tot"), nonlinear=False)

        chisq = 0.0
        chi2_hst = 0.0   # NEW
        chi2_jwst = 0.0  # NEW
                
        for z in self.zs_all:
            rows = data[data[:, 0] == z, 1:]
            if rows.size == 0:
                continue

            HMFs = self.HMF_all(z, Pk_interp, 
                                self.Anorm, 
                                self.qnorm, 
                                self.cnorm
                               )

            Hz = self.provider.get_Hubble(z, units='1/Mpc')
            
            MUV_avs = self.MUV_from_Mh_vec(
                z, self.Mhalos, Hz,
                params['alphastar_icept'], params['betastar'],
                params['epsilonstar_icept'], params['Mc_icept'],
                params['alphastar_slope'], params['epsilonstar_slope'], params['Mc_slope'],
                self.kappaUV, self.invMpctoinvYear,
                )

            MUVs   = rows[:, 0][:, None]
            widths = rows[:, 1][:, None]
            phi    = rows[:, 2]
            sig_up = rows[:, 3]
            sig_lo = rows[:, 4]

            # --- choose scatter per survey ---
            sigma_MUV_this = sigma_MUV_hst if (z < 8.1) else sigma_MUV_jwst
    
            FI    = self.first_integrand(MUVs.T, widths.T, MUV_avs[:, None], sigma_MUV_this)
            preds = self.wM @ (HMFs[:, None] * FI / widths.T)

            use_up = preds > phi
            sigma = np.where(use_up, sig_up, sig_lo)
            # compute this-z contribution once
            chi2_z = float(np.sum(((preds - phi) / sigma) ** 2))

            # add to totals
            chisq += chi2_z
            if z < 8.1:
                chi2_hst += chi2_z
            else:
                chi2_jwst += chi2_z

        loglkl = -0.5 * chisq
        # return per-survey chi2 as derived params
        # params_values["_derived"]["hst"] = chi2_hst
        # params_values["_derived"]["jwst"] = chi2_jwst
        # derived = {"hst": float(chi2_hst), "jwst": float(chi2_jwst)}
        # params_values["_derived"]["sigma_12"] = sigma_12
        if _derived is not None:
            _derived["chi2_hst"] = float(chi2_hst)
            _derived["chi2_jwst"] = float(chi2_jwst)
            
        return loglkl #, derived
        
    # ---------- Helpers ----------
    def parse_assign(self, path_to_data_file, structure):
        try:
            with open(path_to_data_file, 'r') as f:
                for line in f:
                    if not line:
                        continue
                    lhs = line.split('=')[0]
                    if lhs.find(structure + '.') != -1:
                        line = line.replace(structure + '.', '', 1)
                        attr = line.split('=')[0].strip()
                        val = float(line.split('=')[1].split('#')[0])
                        setattr(self, attr, val)
        except IOError:
            print(f'file not found: {path_to_data_file}')
