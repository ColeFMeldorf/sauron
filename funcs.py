

# Standard Library

import numpy as np
import logging
from scipy.stats import binned_statistic as binstat
from astropy import units as u
from scipy.stats import chi2 as chi2_dist
from scipy.special import erfinv
from scipy.integrate import quad


def chi2_unsummed(x, null_counts, f_norm, z_centers, eff_ij, n_data, rate_function, cov_sys=0):
    zJ = z_centers
    fJ = rate_function(zJ, x)
    Ei = np.sum(null_counts * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(null_counts * eff_ij * f_norm**2 * fJ**2, axis=0)

    cov_stat = np.diag(var_Ei + var_Si)
    if cov_sys is None:
        cov_sys = 0
    cov = cov_stat + cov_sys

    inv_cov = np.linalg.pinv(cov)

    resid_matrix = np.outer(n_data - Ei, n_data - Ei)
    chi_squared = np.sum(inv_cov * resid_matrix, axis=0)

    #ones_matrix = np.ones_like(cov)
    #weights = np.sum(inv_cov * ones_matrix, axis=0)
    #logging.debug(f"Weights: {weights}")

    # The difference between the above and below actually matters even though I think it shouldn't.

    resid_vector = n_data - Ei
    chi_squared = resid_vector.T * inv_cov @ resid_vector

    # This vector is X^2 contribution for each z bin. It has ALREADY been squared.
    # I believe the minimizer wants the unsquared version, but it is minimizing the same thing
    # either way I believe.
    # chi = np.sqrt(chi_squared)

    chi = np.sqrt(np.abs(chi_squared))

    #return chi_squared
    return chi

def chi2(x, null_counts, f_norm, z_centers, eff_ij, n_data, rate_function, cov_sys=0):
    zJ = z_centers
    fJ = rate_function(zJ, x)
    Ei = np.sum(null_counts * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(null_counts * eff_ij * f_norm**2 * fJ**2, axis=0)

    cov_stat = np.diag(var_Ei + var_Si)
    if cov_sys is None:
        cov_sys = 0
    cov = cov_stat + cov_sys

    inv_cov = np.linalg.pinv(cov)

    resid_vector = n_data - Ei

    # Note this multiplication is different from the unsummed version above.
    chi_squared = resid_vector.T @ inv_cov @ resid_vector

    # This vector is X^2 contribution for each z bin. It has ALREADY been squared.
    # I believe the minimizer wants the unsquared version, but it is minimizing the same thing
    # either way I believe.
    # chi = np.sqrt(chi_squared)

    return chi_squared


def calculate_covariance_matrix_term(sys_func, sys_params, z_bins, *args):
    # Calculate covariance matrix term for a given systematic function and its parameters
    # sys_func should be a function that takes sys_params and returns expected counts
    # args are additional arguments needed for sys_func.
    logging.info("Calculating Covariance Matrix Term...")
    expected_counts = []
    for i, param in enumerate(sys_params):
        expected_counts.append(sys_func(param, *args))

    for i, E in enumerate(expected_counts):
        if i == 0:
            fiducial = E
            C_ij = np.zeros((len(E), len(E)))
        else:
            C_ij_term = np.outer(E - fiducial, E - fiducial) * 1/(len(sys_params)-1)  # Fixed this error
            C_ij += C_ij_term

    var = np.diag(C_ij)
    denominator = np.outer(np.sqrt(var), np.sqrt(var))
    denominator[denominator == 0] = 1  # Avoid division by zero

    return C_ij


def rescale_CC_for_cov(rescale_vals, PROB_THRESH, index, survey, datasets, z_bins, cheat):
    if cheat:
        datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)
    else:
        sim_IA = datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH)

        sim_CC_df_no_cut = datasets[f"{survey}_SIM_CC"].df
        sim_CC_df = sim_CC_df_no_cut[datasets[f"{survey}_SIM_CC"].prob_scone() > PROB_THRESH]
        types = sim_CC_df_no_cut.TYPE.unique()
        assert (rescale_vals.shape[0] == len(types)), (
            f"I got the wrong number of rescale values compared to contamination types "
            f"({rescale_vals.shape[0]} vs {len(types)})"
        )

        N_CC_sim = np.zeros((len(z_bins)-1, len(types)))
        sim_CC = np.zeros((len(z_bins)-1, len(types)))

        for i, t in enumerate(types):
            rescale_factor = rescale_vals[i]
            raw_counts = binstat(sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            sim_CC[:, i] = raw_counts

            raw_counts = binstat(sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            N_CC_sim[:, i] = raw_counts

        sim_CC = np.sum(sim_CC, axis=1)
        IA_frac = np.nan_to_num(sim_IA / (sim_IA + sim_CC))
        N_CC_sim = np.sum(N_CC_sim)
        n_CC_sim = np.sum(sim_CC)

        N_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins))
        n_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH))
        R = n_data / N_data

        N_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
        n_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH))

        S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)

        CC_frac = (1 - IA_frac) * S
        IA_frac = np.nan_to_num(1 - CC_frac)

        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac

    return n_data


def power_law(z, x):
    alpha, beta = x
    return alpha * (1 + z)**beta


def turnover_power_law(z, x):
    alpha1, beta1, beta2 = x
    z_turn = 1
    alpha2 = alpha1 * (1 + z_turn)**(beta1 - beta2) # Ensure continuity at z_turn
    fJ = np.where(z < z_turn,
                  alpha1 * (1 + z)**beta1,
                  alpha2 * (1 + z)**beta2)
    return fJ


import astropy.cosmology as cosmo
cosmology = cosmo.LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def cosmic_SFR(z,a,b,c,d):
    H0 = cosmology.H0.to("km/s*Mpc").value  # in km/s/Mpc
    return (a + b * z) / (1 + (z / c)**d) * H0 / 100


def cosmic_SFR_dt_dz(z, a, b, c, d):
    H0 = cosmology.H0  # in km/s/Mpc
    Om0 = cosmology.Om0
    Ode0 = cosmology.Ode0
    dt_dz = 1 / H0 / (1 + z) / np.sqrt(Ode0 + Om0 * (1 + z)**3)  # in Gyr per unit redshift
    dt_dz = dt_dz.to("yr").value  # convert to years

    return cosmic_SFR(z, a, b, c, d) * dt_dz


def cosmic_SFR_integrated(z, a, b, c, d):
    return quad(cosmic_SFR_dt_dz, 0, z, args=(a, b, c, d))[0]


def AplusB_cosmicSFH(z, x):
    """ A + B model with cosmic star formation history """
    # Currently using values from Dilday 2008 but should be updated.
    a = 0.0118
    b = 0.08
    c = 3.3
    d = 5.2

    A,B = x

    rho_dot_evaluated = cosmic_SFR(z, a, b, c, d)
    vec_rho_integrated = np.vectorize(cosmic_SFR_integrated)
    rho_integrated_evaluated = vec_rho_integrated(z, a, b, c, d)
    return A * rho_integrated_evaluated + B * rho_dot_evaluated


def calculate_null_counts(z_bins, z_centers, N_gen=None, true_rate_function=None, rate_params=None,
                          time=None, solid_angle=None, cosmo=None):
    """Calculate the number of expected counts for 1 SN / Mpc^3 / yr over the survey volume and time."""

    # Method 1, stupid method, divide N_gen by true rate.
    if all(v is not None for v in [N_gen, true_rate_function, rate_params]):
        fJ = true_rate_function(z_centers, rate_params)
        total_counts = N_gen / fJ
        return total_counts

    # Method 2, harder but more robust method, actually calculate directly from survey parameters.
    logging.info("Calculating null counts via direct integration...")
    logging.info(f"Using time: {time}, solid angle: {solid_angle}")
    logging.info(f"N_gen: {N_gen}")
    if all(v is not None for v in [time, solid_angle]):
        total_counts = []
        for i in range(len(z_centers)):
            z_min = z_bins[i]
            z_max = z_bins[i+1]
            count_sum = SNcount_model(z_min, z_max, RATEPAR={}, genz_wgt=lambda z, par: 1,
                                      HzFUN_INFO=None, SOLID_ANGLE=solid_angle, GENRANGE_PEAKMJD=time, cosmo=cosmo)
            total_counts.append(count_sum.value)

    return np.array(total_counts)



def SNcount_model(zMIN, zMAX, RATEPAR, genz_wgt, HzFUN_INFO, SOLID_ANGLE, GENRANGE_PEAKMJD, cosmo):
    """
    Python translation of the C function SNcount_model.
    FULL DISCLOSURE: This function was created by an AI language model (ChatGPT) 
    based on the provided C code and documentation,
    which was then modified by Cole, because Cole has not used C since middle school.
     However, testing it against the SNANA it seems to give consistent results
    to 1 - 2 sigma with the actual counts that end up in the dump files of SNANA. 
    Since those are slightly stochastic, this is 
    probably acceptable. My fear is that the bias is of the order ~0.1%, which could be an issue 
    when we want to measure rates to
    that precision. But for now, this should be sufficient for testing and development purposes.

    Computes the expected number of SNe between redshifts zMIN and zMAX.

    Parameters
    ----------
    zMIN, zMAX : float
        Redshift integration bounds.

    RATEPAR : object or dict
        Parameters needed by genz_wgt(z, RATEPAR).

    dVdz : callable
        Function dVdz(z, HzFUN_INFO) returning comoving volume element per unit redshift.

    genz_wgt : callable
        Function genz_wgt(z, RATEPAR) giving rate * reweighting factor.

    SOLID_ANGLE : float
        Survey solid angle (same as INPUTS.SOLID_ANGLE in C).

    GENRANGE_PEAKMJD : array-like of length 2
        [MJD_min, MJD_max] — time window.

    Returns
    -------
    float
        Expected number of supernovae.
    """

    # number of integration bins
    NBZ = int((zMAX - zMIN) * 1000.0)
    if NBZ < 10:
        NBZ = 10

    dz = (zMAX - zMIN) / NBZ
    SNsum = 0.0

    # Integration loop (midpoint rule)
    for iz in range(1, NBZ + 1):
        ztmp = zMIN + dz * (iz - 0.5)

        dVdz = cosmo.differential_comoving_volume(ztmp).to(u.Mpc**3 / u.sr).value

        vtmp = dVdz
        rtmp = genz_wgt(ztmp, RATEPAR)

        tmp = rtmp * vtmp / (1.0 + ztmp)
        SNsum += tmp

    # Solid angle and time window
    dOmega = SOLID_ANGLE
    delMJD = GENRANGE_PEAKMJD[1] - GENRANGE_PEAKMJD[0]
    Tyear = delMJD / 365.0

    SNsum *= (dOmega * Tyear * dz)

    return SNsum


def chi2_to_sigma(chi2_diff, dof):
    """Convert chi2 difference to sigma level."""
    # Get p-value from chi2 difference:
    # That's 1 minus the integral of the chi2 distribution from 0 to chi2_diff
    p_value = 1 - chi2_dist.cdf(chi2_diff, dof)

    # Solve for x sigma in terms of p-value:
    # 2 * p = 1 - erf(x / sqrt(2))
    # erf(x / sqrt(2)) = 1 - 2 * p
    # x / sqrt(2) = erfinv(1 - 2 * p)
    # x = sqrt(2) * erfinv(1 - 2 * p)

    sigma = np.sqrt(2) * erfinv(1 - 2 * p_value)
    return sigma
