import logging

import numpy as np
import scipy

from astropy.cosmology import Planck18 as cosmology

from astropy import units as u
from scipy.integrate import cumulative_trapezoid

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


def csfr_double_power_law(z, uncertainty=None, uncertainty_mode="upper"):
    """The cosmic star formation rate density (CSFR), as a double power-law in log10 space.
    Returns solMass yr⁻¹ Mpc⁻³ (float array).

    This is the form that used to be hardcoded as the only available CSFR (it was previously named
    `_CSFR`, called directly from dtd_rate_vec). It's now just one entry in
    `csfr_func_name_dictionary` below, selectable by name from the CSFR field in FIT_OPTIONS.

    NOTE THAT UNCERTAINTY IS EXPECTED TO BE IN DEX, LIKE IN BEHROOZI ET AL.

    Inputs:
    - z: redshift (float or array)
    - uncertainty: function, a function that acts on z and returns uncertainity in dex
    - uncertainty_mode: "upper" or "lower", determines whether to multiply or divide by the uncertainty factor.

    """
    z0, A, B = 1.243, -0.997, 0.241
    C = 0.180

    sfh = C / (10**(A*(z-z0)) + 10**(B*(z-z0)))
    if uncertainty != None:
        if uncertainty_mode == "upper":
            sfh *= 10**(uncertainty(z))
        elif uncertainty_mode == "lower":
            sfh /= 10**(uncertainty(z))
        else:
            raise ValueError("Invalid uncertainty_mode. Use 'upper' or 'lower'.")
    return sfh


def _rational_dbl_pwr_law(z, A, B, C, D):

    numerator = A * (1+z) ** C
    denominator = (((1 + z)/B)**D) + 1
    return numerator / denominator


def strolger_CSFR(z, uncertainty=None, uncertainty_mode="upper"):
    return _rational_dbl_pwr_law(z, A=0.0134, B=2.55, C=3.3, D=6.1)


def precompute_AplusB(z_data, csfr, cosmology, a=0.0118, b=0.08, c=3.3, d=5.2, R=0.56, z_max=100.0, n_grid=10_000):
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

    def _sfr_dt_dz(z_):
        E_z   = np.sqrt(Ode0 + Om0 * (1 + z_) ** 3)   # dimensionless Hubble factor E(z)
        dt_dz = 1.0 / (H0_per_yr * (1 + z_) * E_z)    # yr per unit redshift
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

    rho_integrated = (1 - R) * (total - np.interp(z_data, z_grid, cumul))
    rho_dot        = csfr(z_data)

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
    "S20": strolger_CSFR
}


# def _SNR(t_gyr, eta_Ia, fP):
#     bins = np.linspace(0, 10, 10)
#     indices = np.digitize(t_gyr, bins) - 1
#     indices = np.clip(indices, 0, len(bins) - 2)
#     vals = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]) * 1e-3
#     rate = vals[indices]

#     return rate


def _dt_dz(zprime):
    """dt/dz in yr (float array)."""
    return 1.0 / (_H0_per_yr * (1+zprime) * np.sqrt(_Ode0 + _Om0*(1+zprime)**3))

# ── 3. FULLY VECTORISED fP_rate ───────────────────────────────────────────────
def dtd_rate_vec(z_array, dtd_func, args, kwargs, n_steps=1000, csfh_unc=None, csfh_unc_mode="upper",
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
        raise ValueError("No CSFR function provided.")

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
    return rate * eff


def binned_DTD(t_gyr, *vals, bins = np.linspace(0, 5, 11)):
    indices = np.digitize(t_gyr, bins) - 1
    indices = np.clip(indices, 0, len(vals) - 1)
    vals = np.array(vals)
    rate = vals[indices]
    return rate