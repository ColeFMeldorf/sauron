import logging

import numpy as np
import scipy

#from astropy.cosmology import LambdaCDM
#cosmology = LambdaCDM(H0=70, Om0=0.315, Ode0=0.685)
from astropy.cosmology import Planck18 as cosmology

from astropy import units as u
from scipy.integrate import cumulative_trapezoid


from scipy.optimize import nnls

logger = logging.getLogger(__name__)


# ── 1. PRECOMPUTE COSMOLOGY LOOKUPS (do once, not per z) ──────────────────────
_z_grid = np.linspace(0, 15, 10_000)
_t_grid = cosmology.age(_z_grid).to("Gyr").value          # Gyr, decreasing with z


def fast_age(z):
    """Interpolated age in Gyr — replaces cosmology.age()."""
    return np.interp(z, _z_grid, _t_grid)


def fast_z_at_age(t_gyr):
    """Interpolated redshift at age t_gyr — replaces z_at_value(cosmology.age, t)."""
    # _t_grid is *decreasing*, so flip both arrays for np.interp
    return np.interp(t_gyr, _t_grid[::-1], _z_grid[::-1])

# ── 2. UNIT-STRIPPED KERNELS ──────────────────────────────────────────────────
# Strip astropy units once up-front; add them back on output.

_H0_per_yr  = cosmology.H0.to("1/yr").value          # yr⁻¹
_Om0        = cosmology.Om0
_Ode0       = cosmology.Ode0
_eta_Ia_val = 1.6e-4                                   # (solMass · yr)⁻¹  ← strip unit here


def csfr_double_power_law_uncorrected(z, uncertainty=None, uncertainty_mode="upper"):
    """The cosmic star formation rate density (CSFR), as a double power-law in log10 space.
    Returns solMass yr⁻¹ Mpc⁻³ (float array).

    NOTE THAT UNCERTAINTY IS EXPECTED TO BE IN DEX, LIKE IN BEHROOZI ET AL.

    Inputs:
    - z: redshift (float or array)
    - uncertainty: function, a function that acts on z and returns uncertainity in dex
    - uncertainty_mode: "upper" or "lower", determines whether to multiply or divide by the uncertainty factor.

    """
    z0, A, B = 1.243, -0.997, 0.241
    C = 0.180

    sfh = C / (10**(A*(z-z0)) + 10**(B*(z-z0)))
    if uncertainty is not None:
        if uncertainty_mode == "upper":
            sfh *= 10**(uncertainty(z))
        elif uncertainty_mode == "lower":
            sfh /= 10**(uncertainty(z))
        else:
            raise ValueError("Invalid uncertainty_mode. Use 'upper' or 'lower'.")
    return sfh


def csfr_double_power_law(z, uncertainty=None, uncertainty_mode="upper"):
    return csfr_double_power_law_uncorrected(z, uncertainty=uncertainty, uncertainty_mode=uncertainty_mode) / 0.7


def _rational_dbl_pwr_law(z, A, B, C, D):

    numerator = A * (1+z) ** C
    denominator = (((1 + z)/B)**D) + 1
    return numerator / denominator


def strolger_CSFR(z, uncertainty=None, uncertainty_mode="upper"):
    return _rational_dbl_pwr_law(z, A=0.0134, B=2.55, C=3.3, D=6.1)


def _log_sfr_li(z, a, b):
    log_sfr = a + b * np.log10(1 + z)
    return 10 ** log_sfr


def li_piecewise(z, uncertainty=None, uncertainty_mode="upper"):
    """Li piecewise CSFR from Li (2008)"""
    z = np.asarray(z)
    sfr = np.zeros_like(z)

    sfr[z < 0.993] = _log_sfr_li(z[z < 0.993], a=-1.70, b=3.30)
    sfr[(z >= 0.993) & (z < 3.8)] = _log_sfr_li(z[(z >= 0.993) & (z < 3.8)], a=-0.727, b=0.0549)
    sfr[z >= 3.8] = _log_sfr_li(z[z >= 3.8], a=2.35, b=-4.46)

    return sfr


def precompute_AplusB(z_data, csfr, cosmology,  z_max=100.0, n_grid=10_000):
    """ Pre-computes the SFR terms for fixed cosmology and SFH shape, returning a
    fast callable for fitting. Call this ONCE before fitting.

    The returned function signature matches AplusB_cosmicSFH(z, x) where x = [A, B].
    """
    z_data = np.asarray(z_data)

    # --- Pull everything out of astropy ONCE ---
    H0_val    = cosmology.H0.to("km/s/Mpc").value
    Mpc_in_km = (1 * u.Mpc).to(u.km).value   # 3.0857e19
    yr_in_s   = (1 * u.yr ).to(u.s  ).value   # 3.1557e7
    H0_per_yr = H0_val / Mpc_in_km * yr_in_s  # H0 expressed in yr^-1
    Om0       = float(cosmology.Om0)
    Ode0      = float(cosmology.Ode0)


    # def csfr(z):
    #     a = .0157
    #     b = .118
    #     c = 3.23
    #     d = 4.66
    #     return (a + b * z) / (1 + (z/c)**d)

    def _sfr_dt_dz(z_):
        # E_z   = np.sqrt(Ode0 + Om0 * (1 + z_) ** 3)   # dimensionless Hubble factor E(z)
        # * E_z)
        H_z = cosmology.H(z_).to("1/yr").value
        # H0_per_yr
        dt_dz = 1.0 / (H_z * (1 + z_))     # yr per unit
        return csfr(z_) * dt_dz

    # --- One cumulative integration pass replaces N separate quad calls ---
    # Instead of integrating from each z_i to 100 independently, we compute
    # the cumulative integral from 0 → z_max once on a fine grid, then
    # use the identity:
    #   integral(z_i → z_max) = total - integral(0 → z_i)
    # and interpolate at the actual data z values.

    z_grid    = np.linspace(0.0, z_max, n_grid)
    cumul     = cumulative_trapezoid(_sfr_dt_dz(z_grid), z_grid, initial=0.0)
    total     = cumul[-1]
    # (1 - R) *
    # (0.7) ** 3 *
    rho_integrated =  (total - np.interp(z_data, z_grid, cumul))
    rho_dot = csfr(z_data)

    np.save("rho_integrated.npy", rho_integrated)
    np.save("rho_dot.npy", rho_dot)

    # from matplotlib import pyplot as plt
    # plt.plot(z_data, rho_integrated, label="rho_integrated")
    # # manually take the derivative of rho integrated
    # rho_integrated_deriv = np.gradient(rho_integrated, z_data)
    # dz_dt = cosmology.H(z_data).to("1/yr").value * (1 + z_data)
    # rho_integrated_deriv_dt = rho_integrated_deriv * dz_dt
    # plt.plot(z_data, -1 * rho_integrated_deriv_dt, label="deriv of rho_integrated")
    # print("rho deriv integrated: ", -1 * rho_integrated_deriv_dt)
    # print("rho dot: ", rho_dot)

    # plt.plot(z_data, rho_dot, label="rho_dot")
    # plt.legend()
    # plt.xlabel("z")
    # plt.ylabel("SFR terms")
    # plt.title("Precomputed SFR terms for A+B model")
    # plt.yscale("log")
    # plt.savefig("precomputed_sfr_terms.png")
    # plt.close()

    # --- This is all that runs during fitting: two multiplications ---
    def AplusB_fast(z, x):
        A, B = x
        return A * rho_integrated + B * rho_dot

    def AplusB_fast_interp(z, x):
        f = np.interp(z, z_data, AplusB_fast(z_data, x))
        return f

    return AplusB_fast_interp

# Registry of selectable CSFR functions — same idea as dtd_func_name_dictionary / func_name_dictionary
# in runner.py. To add a new CSFR functional form: write a function with the same signature as
# csfr_double_power_law (z, uncertainty=None, uncertainty_mode="upper") and add it here under whatever
# name you want to refer to it by in the config file's CSFR field.


csfr_func_name_dictionary = {
    "B13": csfr_double_power_law,
    "B13_uncorrected": csfr_double_power_law_uncorrected,
    "S20": strolger_CSFR,
    "L08": li_piecewise
}



def _dt_dz(zprime):
    """dt/dz in yr (float array)."""
    return 1.0 / (_H0_per_yr * (1+zprime) * np.sqrt(_Ode0 + _Om0*(1+zprime)**3))

# ── 3. FULLY VECTORISED fP_rate ───────────────────────────────────────────────
def dtd_rate_vec(z_array, dtd_func, args, kwargs, n_steps=10000, csfh_unc=None, csfh_unc_mode="upper",
                  csfr_func=None):
    """ Evaluate fP_rate for every z in z_array simultaneously.

    Parameters
    ----------
    z_array : array-like  shape (N,)
    dtd_func: function that represents the Delay Time Distribution (DTD).
        All DTD functions must have arguments in the form:
        (t_gyr, parameters to be fit (pos), set parameters (kwd))
    csfr_func : function, optional
        The cosmic star formation rate function to convolve the DTD with. Must accept
        (z, uncertainty=None, uncertainty_mode="upper"), same as csfr_double_power_law. Defaults to
        csfr_double_power_law if not given, so existing callers that don't pass this keep working
        exactly as before.


    Returns
    -------
    rates : ndarray shape (N,)  in yr⁻² Mpc⁻³
    """
    if csfr_func is None:
        csfr_func = csfr_double_power_law

    z_array = np.asarray(z_array, dtype=float)
    N = len(z_array)

    # --- integration bounds (all N redshifts at once) ---
    t_now = fast_age(z_array)  # (N,)  Gyr

    # Enforce a minimum delay of 0.04 Gyr (40 Myr) and ensure the integration lower bound is always earlier than t_now.
    t_high = t_now - 0.04
    t_low = np.maximum(t_high - 10.0, 0.0)

    # --- 2-D time grid: shape (N, n_steps) ---
    # Row i spans  t_high[i] → t_low[i]  (decreasing in time = increasing in z)
    alpha = np.linspace(0, 1, n_steps)  # (n_steps,)
    t_grid = t_high[:, None] + alpha[None, :] * (t_low[:, None] - t_high[:, None])

    # --- corresponding redshifts  (single vectorised call) ---
    z_grid = fast_z_at_age(t_grid.ravel()).reshape(N, n_steps)

    # --- delay times Δt = t_now − t′  (N, n_steps) ---
    delta_t = t_now[:, None] - t_grid  # Gyr, ≥ 0.04

    # --- evaluate the three factors ---
    csfr    = csfr_func(z_grid, uncertainty=csfh_unc, uncertainty_mode=csfh_unc_mode)  # (N, n_steps)
    snr     = arbitrary_DTD(dtd_func, delta_t, args, kwargs)                   # (N, n_steps)
    dt_dz   = _dt_dz(z_grid)                                 # (N, n_steps), yr

    integrand = csfr * snr * dt_dz                           # (N, n_steps)

    # --- integrate each row over its own z′ axis ---
    rates = scipy.integrate.trapezoid(integrand, z_grid, axis=1)  # (N,)
    rates *= 1e-9
    return rates   # units: solMass yr⁻¹ Mpc⁻³ · (solMass yr Gyr)⁻¹ · yr = yr⁻² Mpc⁻³ Gyr⁻¹
                   # multiply by 1e-9 to convert the Gyr⁻¹ → yr⁻¹ if needed


def dtd_rate(dtd_func, kwargs=None, n_steps=1000, csfh_unc=None, csfh_unc_mode="upper", csfr_func=None):
    """ Define a wrapper function that allows us to set the DTD function (and the CSFR it's convolved
    with) once and then call that repeatedly during fitting. """
    if kwargs is None:
        kwargs = {}
    return lambda z, args: dtd_rate_vec(z, dtd_func, args, kwargs, n_steps, csfh_unc, csfh_unc_mode,
                                        csfr_func=csfr_func)


def arbitrary_DTD(func, t_gyr, args, kwargs):
    return func(t_gyr, *args, **kwargs)


# All DTD functions must have arguments in the form: (t_gyr, parameters to be fit (pos), set parameters (kwd))

def power_law_DTD(t_gyr, beta, eff, cutoff=0.01):
    # Enforce a minimum delay time to avoid divergences at small t.
    t = np.asarray(t_gyr, dtype=float)
    t = np.maximum(t, cutoff)
    rate = t**beta
    eff *= 1e9 # convert from Gyr^-1 to yr^-1
    return rate * eff


def binned_DTD(t_gyr, *vals, bins = np.linspace(0, 5, 11)):
    indices = np.digitize(t_gyr, bins) - 1
    indices = np.clip(indices, 0, len(vals) - 1)
    vals = np.array(vals)
    rate = vals[indices]
    return rate

def prompt_fraction_DTD(t, eta_Ia, fP):
    K = 7.132
    # t is measured in Gyr
    rate = np.zeros_like(t)
    rate[t < 0.04] = 0
    rate[(0.04 <= t) & (t < 0.5)] = eta_Ia * fP * K / (1 - fP)
    rate[t >= 0.5] = eta_Ia * t[t >= 0.5]**-1

    return rate


# ----------------------------------------------------------------------
# 1. Helper: redshift -> cosmic age (Gyr)
# ----------------------------------------------------------------------
def age_at_z(z):
    """Age of the universe (Gyr) at redshift z. Vectorized."""
    return cosmology.age(np.atleast_1d(z)).to(u.Gyr).value


# ----------------------------------------------------------------------
# 2. Build the response matrix A such that  R_sn = A @ Phi
# ----------------------------------------------------------------------
def build_response_matrix(z_sn_edges, z_csfr_edges, psi_csfr_peryr, tau_edges_Gyr):
    """Parameters
    ----------
    z_sn_edges : array (n_sn+1,)
        Redshift bin EDGES for the SN Ia rate measurement (coarse).
    z_csfr_edges : array (n_csfr+1,)
        Redshift bin EDGES for the CSFR (fine grid).
    psi_csfr_peryr : array (n_csfr,)
        CSFR density per fine bin, in Msun / yr / Mpc^3
        (i.e. the usual Madau & Dickinson-style units).
    tau_edges_Gyr : array (n_dtd+1,)
        Delay time bin edges in Gyr, e.g. [0.0, 0.5, 2.0, 13.5] -> 3 bins.

    Returns
    -------
    A : array (n_sn, n_dtd)
        Response matrix, units such that A @ Phi_[SNe/Msun/Gyr] gives
        R_sn in SNe / Gyr / Mpc^3.
    t_sn_center : array (n_sn,)
        Cosmic age (Gyr) at the center of each SN redshift bin.
    """
    # --- convert redshift edges to cosmic time (age) ---
    t_sn_edges = age_at_z(z_sn_edges)          # decreasing as z increases
    t_csfr_edges = age_at_z(z_csfr_edges)

    t_sn_center = 0.5 * (t_sn_edges[:-1] + t_sn_edges[1:])

    t_csfr_center = 0.5 * (t_csfr_edges[:-1] + t_csfr_edges[1:])
    dt_csfr = np.abs(t_csfr_edges[1:] - t_csfr_edges[:-1])   # Gyr

    # convert CSFR to per-Gyr so units match dt_csfr (Gyr)
    psi_csfr_perGyr = psi_csfr_peryr * 1e9   # Msun/yr/Mpc^3 -> Msun/Gyr/Mpc^3

    # check if any t_sn_edges are out of order
    out_of_order = 0
    for i in range(len(t_sn_edges) - 1):
        if t_sn_edges[i] < t_sn_edges[i + 1]:
            out_of_order += 1
            logging.debug("out of order bins: " + str(t_sn_edges[i]) + " < " + str(t_sn_edges[i + 1]))
    n_sn = len(t_sn_center) - out_of_order
    n_dtd = len(tau_edges_Gyr) - 1
    A = np.zeros((n_sn, n_dtd))

    for i, t_i in enumerate(t_sn_center):
        if t_sn_edges[i] > t_sn_edges[i + 1]:
            # We want to skip cases where the SN bin loops back to start, aka between different surveys
            continue
        # delay (Gyr) between each fine CSFR bin and this SN epoch
        tau_k = t_i - t_csfr_center
        for j in range(n_dtd):
            tau_lo, tau_hi = tau_edges_Gyr[j], tau_edges_Gyr[j + 1]
            # only mass formed BEFORE the SN epoch (tau_k >= 0) can contribute
            in_bin = (tau_k >= tau_lo) & (tau_k < tau_hi) & (tau_k >= 0)
            A[i, j] = np.sum(psi_csfr_perGyr[in_bin] * dt_csfr[in_bin])

    return A, t_sn_center


# ----------------------------------------------------------------------
# 3. Solve the linear system for Phi (the binned DTD)
# ----------------------------------------------------------------------
def recover_dtd(R_sn_peryr, A, nonnegative=True):
    """
    R_sn_peryr : array (n_sn,), SN Ia rate in SNe / yr / Mpc^3
    A          : response matrix from build_response_matrix
    Returns Phi in SNe / Msun / Gyr, one value per delay bin.
    """
    R_sn_perGyr = R_sn_peryr * 1e9   # match the per-Gyr convention used in A

    if nonnegative:
        logging.debug("A size: " + str(A.shape))
        logging.debug("R_sn_perGyr size: " + str(R_sn_perGyr.shape))
        Phi, resid = nnls(A, R_sn_perGyr)
    else:
        Phi, *_ = np.linalg.lstsq(A, R_sn_perGyr, rcond=None)
    return Phi


def calculate_DTD_x0_vals(z_sn_edges, csfr_func_name, tau_edges_Gyr, VSNR):

    z_csfr_edges = np.linspace(0.001, 4.0, 400)  # fine grid, edges
    z_csfr_center = 0.5 * (z_csfr_edges[:-1] + z_csfr_edges[1:])

    psi_csfr = csfr_func_name_dictionary[csfr_func_name](z_csfr_center)
    logger.debug("z_sn_edges: " + str(z_sn_edges))
    logger.debug("z_sn_edges size: " + str(len(z_sn_edges)))
    logger.debug("VSNR size: " + str(VSNR.shape))
    logger.debug("phi csfr size: " + str(psi_csfr.shape))
    logger.debug("tau edges size: " + str(tau_edges_Gyr.shape))
    logger.debug("z_csfr_edges size: " + str(z_csfr_edges.shape))

    A, t_sn_center = build_response_matrix(
        z_sn_edges, z_csfr_edges, psi_csfr, tau_edges_Gyr
    )
    recovered_Phi = recover_dtd(VSNR, A, nonnegative=True)

    logger.debug("Recovered DTD values (SNe / Msun / Gyr): {}".format(recovered_Phi))

    return recovered_Phi