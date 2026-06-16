import numpy as np
import scipy
import astropy.units as u

from astropy.cosmology import Planck18 as cosmology
from astropy.cosmology import z_at_value
from astropy import units as u

import logging
logger = logging.getLogger(__name__)

H0 = cosmology.H0.to("km/s*Mpc") # in km/s/M
Om0 = cosmology.Om0
Ode0 = cosmology.Ode0


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

def _CSFR(z, uncertainty = None, uncertainty_mode = "upper"):
    """Returns solMass yr⁻¹ Mpc⁻³ (float array).

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
def dtd_rate_vec(z_array, dtd_func, args, kwargs, n_steps=1000, csfh_unc = None, csfh_unc_mode = "upper"):
    """ Evaluate fP_rate for every z in z_array simultaneously.

    Parameters
    ----------
    z_array : array-like  shape (N,)
    dtd_func: function that represents the Delay Time Distribution (DTD).
        All DTD functions must have arguments in the form:
        (t_gyr, parameters to be fit (pos), set parameters (kwd))


    Returns
    -------
    rates : ndarray shape (N,)  in yr⁻² Mpc⁻³
    """
    z_array = np.asarray(z_array, dtype=float)
    N = len(z_array)

    # --- integration bounds (all N redshifts at once) ---
    t_now  = fast_age(z_array)                               # (N,)  Gyr
    t_low  = np.maximum(t_now - 10.0, 1.0)                  # (N,)  Gyr


    # --- 2-D time grid: shape (N, n_steps) ---
    # Row i spans  t_now[i] → t_low[i]  (decreasing in time = increasing in z)
    alpha  = np.linspace(0, 1, n_steps)                      # (n_steps,)
    t_grid = t_now[:, None] + alpha[None, :] * (t_low[:, None] - t_now[:, None]) - 0.04
    # shape (N, n_steps),  Gyr

    # --- corresponding redshifts  (single vectorised call) ---
    z_grid = fast_z_at_age(t_grid.ravel()).reshape(N, n_steps)

    # --- delay times Δt = t_now − t′  (N, n_steps) ---
    delta_t = t_now[:, None] - t_grid                        # Gyr, ≥ 0.04 (40 Myr, min WD formation time)


    # --- evaluate the three factors ---
    csfr    = _CSFR(z_grid, uncertainty=csfh_unc, uncertainty_mode=csfh_unc_mode)  # (N, n_steps)
    snr     = arbitrary_DTD(dtd_func, delta_t, args, kwargs)                   # (N, n_steps)
    dt_dz   = _dt_dz(z_grid)                                 # (N, n_steps), yr

    integrand = csfr * snr * dt_dz                           # (N, n_steps)

    # --- integrate each row over its own z′ axis ---
    rates = scipy.integrate.trapezoid(integrand, z_grid, axis=1)  # (N,)
    rates *= 1e-9
    return rates   # units: solMass yr⁻¹ Mpc⁻³ · (solMass yr Gyr)⁻¹ · yr = yr⁻² Mpc⁻³ Gyr⁻¹
                   # multiply by 1e-9 to convert the Gyr⁻¹ → yr⁻¹ if needed


def dtd_rate(dtd_func, kwargs=None, n_steps=1000, csfh_unc = None, csfh_unc_mode = "upper"):
    """ Define a wrapper function that allows us to set the DTD function once and then call that. """
    if kwargs is None:
        kwargs = {}
    return lambda z, args: dtd_rate_vec(z, dtd_func, args, kwargs, n_steps, csfh_unc, csfh_unc_mode)


def arbitrary_DTD(func, t_gyr, args, kwargs):
    return func(t_gyr, *args, **kwargs)


# All DTD functions must have arguments in the form: (t_gyr, parameters to be fit (pos), set parameters (kwd))

def power_law_DTD(t_gyr, beta, eff, cutoff = 0.01):
    max_t = np.max(t_gyr)
    min_t = np.min(t_gyr)

    original_min_t = min_t
    if min_t < cutoff:
        min_t = cutoff


    #upper = max_t**(beta + 1.0) / (beta + 1.0)
    #lower = min_t**(beta + 1.0) / (beta + 1.0)
    #constant_zone = (cutoff**beta) * (cutoff - original_min_t)

    #area_under_curve = constant_zone + upper - lower


    rate = t_gyr**beta
    #rate[np.where(t_gyr < cutoff)] = cutoff**beta
    #rate /= area_under_curve


    return rate * eff


# def binned_DTD(t_gyr, vals, eff, bins = None):
#     indices = np.digitize(t_gyr, bins) - 1

#     indices = np.clip(indices, 0, len(vals) - 1)

#     rate = vals[indices]

#     area_under_curve = np.sum(vals * np.diff(bins))

#     rate /= area_under_curve
#     rate *= eff
#     return rate

def binned_DTD(t_gyr, *vals, bins = np.linspace(0, 5, 11)):
    indices = np.digitize(t_gyr, bins) - 1
    indices = np.clip(indices, 0, len(vals) - 1)
    vals = np.array(vals)
    rate = vals[indices]
    return rate
