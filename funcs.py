

# Standard Library
import pandas as pd
import numpy as np
import logging
from scipy.stats import binned_statistic as binstat
from astropy import units as u
from scipy.stats import chi2 as chi2_dist
from scipy.special import erfinv

logger = logging.getLogger(__name__)


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

    # The difference between the above and below actually matters even though I think it shouldn't.

    # resid_vector = n_data - Ei
    # chi_squared = resid_vector.T * inv_cov @ resid_vector

    # This vector is X^2 contribution for each z bin. It has ALREADY been squared.
    # I believe the minimizer wants the unsquared version, but it is minimizing the same thing
    # either way I believe.
    # chi = np.sqrt(chi_squared)

    #chi = np.sqrt(np.abs(chi_squared))

    return chi_squared
    #return chi

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
    logger.info("Calculating Covariance Matrix Term...")
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


def rescale_CC_for_cov(rescale_vals_and_seeds, PROB_THRESH, index, survey, datasets, z_bins, cheat):
    if cheat:
        datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)
    else:
        rescale_vals = rescale_vals_and_seeds[:-1]
        seed = int(rescale_vals_and_seeds[-1])
        # The last entry is the seed for reproducibility.
        sim_IA = datasets[f"{survey}_SIM_IA"]
        sim_CC_df_no_cut = datasets[f"{survey}_SIM_CC"].df
        types = sim_CC_df_no_cut.TYPE.unique()
        # Separate the CC SNe by type
        sim_CC_sep_on_type = [sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t] for t in types]
        # Resample each type according to the rescale values
        sizes = np.array([len(df) for df in sim_CC_sep_on_type])
        new_sizes = np.round(sizes * rescale_vals, decimals=0).astype(int)
        new_CC_sep_on_type = [sim_CC_sep_on_type[i].sample(n=new_sizes[i], replace=True, random_state=seed)
                              for i in range(len(types))]
        new_sim_CC_df = pd.concat(new_CC_sep_on_type)

        # Now apply the probability cut. This is doing the CC decontamination.
        scone_col_name = datasets[f"{survey}_SIM_CC"].scone_col
        cc_scone_col = new_sim_CC_df[scone_col_name]
        new_sim_CC_df_cut = new_sim_CC_df[cc_scone_col > PROB_THRESH]
        sim_IA_df_cut = sim_IA.df[sim_IA.df[sim_IA.scone_col] > PROB_THRESH]
        temp_sim_all_df = pd.concat([new_sim_CC_df_cut, sim_IA_df_cut])
        temp_sim_all_cut_counts = binstat(temp_sim_all_df[datasets[f"{survey}_SIM_CC"].z_col],
                                          temp_sim_all_df[datasets[f"{survey}_SIM_CC"].z_col],
                                          statistic='count', bins=z_bins)[0]
        # Finally do the bias correction as in CC decontam in the main analysis.
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
        bias_correction = datasets[f"{survey}_SIM_IA"].z_counts(z_bins) / temp_sim_all_cut_counts
        bias_correction = np.nan_to_num(bias_correction, nan=1.0, posinf=1.0, neginf=1.0)
        n_data *= bias_correction

    return n_data


def power_law(z, x):
    alpha, beta = x
    return alpha * (1 + z)**beta


def turnover_power_law(z, x):
    alpha1, beta1, alpha2, beta2 = x
    z_turn = 1
    fJ = np.where(z < z_turn,
                  alpha1 * (1 + z)**beta1,
                  alpha2 * (1 + z)**beta2)
    return fJ


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
