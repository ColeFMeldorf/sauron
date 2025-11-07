

# Standard Library

import numpy as np
from scipy.stats import binned_statistic as binstat


def chi2(x, N_gen, f_norm, z_centers, eff_ij, n_data, rate_function, cov_sys=0):
    zJ = z_centers
    fJ = rate_function(zJ, x)
    Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=0)

    cov_stat = np.diag(var_Ei + var_Si)
    if cov_sys is None:
        cov_sys = 0
    cov = cov_stat + cov_sys
    inv_cov = np.linalg.pinv(cov)
    resid_matrix = np.outer(n_data - Ei, n_data - Ei)
    chi_squared = np.sum(inv_cov * resid_matrix, axis=0)

    return chi_squared


def calculate_covariance_matrix_term(sys_func, sys_params, z_bins, *args):
    # Calculate covariance matrix term for a given systematic function and its parameters
    # sys_func should be a function that takes sys_params and returns expected counts
    # args are additional arguments needed for sys_func.
    print("Calculating Covariance Matrix Term...")
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
        assert(rescale_vals.shape[0] == len(types)), "I got the wrong number"
        " of rescale values compared to contamination types" \
            f"({rescale_vals.shape[0]} vs {len(types)})"


        N_CC_sim = np.zeros((len(z_bins)-1, len(types)))
        sim_CC = np.zeros((len(z_bins)-1, len(types)))

        for i,t in enumerate(types):
            rescale_factor = rescale_vals[i]
            raw_counts = binstat(sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            sim_CC[:,i] = raw_counts

            raw_counts = binstat(sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                 statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            N_CC_sim[:,i] = raw_counts

        sim_CC = np.sum(sim_CC, axis = 1)


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
        print("Calculated a Ia frac of:", IA_frac)

        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac

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
