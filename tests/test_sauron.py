
# Sauron
from funcs import chi2, calculate_covariance_matrix_term, power_law, calculate_null_counts, \
    rescale_CC_for_cov
from runner import sauron_runner

# Standard Library
import os
import logging
from matplotlib import pyplot as plt
import pathlib
import pytest
from types import SimpleNamespace
import subprocess
from scipy.stats import chi2 as scipy_chi2, ks_2samp, norm


import numpy as np
import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

global_rtol = 5e-3
# We aren't measuring even close to the 0.5% level of precision
warning_rtol = 1e-6
# but I do want a warning if it changed even a little bit.


# # Configure the basic logging setup
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
#     force = True
# )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _coverage_chi2(df, truth, param_names=("alpha", "beta")):
    """Per-row chi-squared-like statistic used by the coverage tests below, to check whether the
    fitted parameters are consistent with the known truth values at the expected rate.

    If df has asymmetric '{param}_error_upper' / '{param}_error_lower' columns for every parameter in
    param_names (i.e. add_results() was called with error_upper/error_lower), a separate covariance
    matrix is built for EACH ROW (each simulated dataset), since which side of the asymmetric error to
    use can differ row to row:
      - diagonal entries: for a given row and parameter, use the lower-side variance if that row's
        residual (fit - truth) is positive (the fit landed above truth, so truth lies below the fit --
        it's the lower error that measures that distance), and the upper-side variance if the residual
        is negative (fit landed below truth -- the upper error measures that distance).
      - off-diagonal entries:
        Now, off diagonal entries are not used at all. I am not sure how to handle covariance for this case.

        Before:
        taken directly from the existing per-row 'cov_{p1}_{p2}' column (each
        simulated dataset already has its own fitted covariance between parameters; that doesn't change
        based on which side of the split normal is used for the diagonal).
    Each row's matrix is inverted individually and the Mahalanobis-distance-squared statistic
    (residual^T @ inv_cov @ residual) is computed per row.

    Otherwise, falls back to the original approach: a single symmetric covariance matrix built from
    the median '{param}_error'**2 values and the median 'cov_{p1}_{p2}' cross term (same matrix for
    every row), inverted once.
    """
    has_split = all(
        f"{p}_error_upper" in df.columns and f"{p}_error_lower" in df.columns
        for p in param_names
    )


    n = len(df)
    k = len(param_names)

    # residuals[:, j] = fitted value of parameter j minus its truth, for every row
    residuals = np.column_stack([df[p].to_numpy() - t for p, t in zip(param_names, truth)])

    # Pick the appropriate one-sided sigma for each row and parameter
    sigmas = np.empty((n, k))
    if has_split:
        for j, p in enumerate(param_names):
            sigma_lower = df[f"{p}_error_lower"].to_numpy()
            sigma_upper = df[f"{p}_error_upper"].to_numpy()
            sigmas[:, j] = np.where(residuals[:, j] > 0, sigma_lower, sigma_upper)
            print("Sigmas from upper / lower:", sigmas[:, j])
            print("On the other hand, had we used the symmetric error:", df[f"{p}_error"].to_numpy())
            print("Ratio:", sigmas[:, j] / df[f"{p}_error"].to_numpy())
    else:
        sigmas = np.column_stack([df[f"{p}_error"].to_numpy() for p in param_names])

    chi2_vals = np.empty(n)
    for i in range(n):
        cov_i = np.diag(sigmas[i] ** 2)
        for a in range(k):
            for b in range(a + 1, k):
                col = f"cov_{param_names[a]}_{param_names[b]}"
                if col in df.columns:
                    if not has_split:
                        cov_i[a, b] = df[col].iloc[i]
                        cov_i[b, a] = df[col].iloc[i]
        inv_cov_i = np.linalg.inv(cov_i)
        chi2_vals[i] = residuals[i] @ inv_cov_i @ residuals[i]
    return chi2_vals


def _check_regression(results_path, regression_path, cols, rtol=None, atol=None):
    if rtol is None:
        rtol = global_rtol
    results = pd.read_csv(results_path)
    regression = pd.read_csv(regression_path)
    for i, col in enumerate(cols):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        logger.debug(f"Checking {col}")
        logger.debug(f"results[col]: {results[col]}")
        logger.debug(f"regression[col]: {regression[col]}")
        logger.debug(f"rtol: {rtol}, atol: {atol}")
        # , rtol=rtol, atol=atol
        np.testing.assert_allclose(results[col], regression[col], rtol=rtol, **({} if atol is None else {"atol": atol}))

def _run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
# ##########################TESTS BELOW############################################################

def test_regression_specz():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_regnopz_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov",
     "--prob_thresh", "0.5", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    cols = ["alpha", "beta", "reduced_chi_squared"]
    _check_regression(outpath, pathlib.Path(__file__).parent / "test_regression/test_regnopz_regression.csv", cols)



def test_regression_pz_5datasets():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Photo Zs.
       This also uses 5 datasets rather than 1 to test that functionality.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_regpz_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_5pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov", "--prob_thresh", "0.5", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/test_regpz_regression.csv")
    for i, col in enumerate(regression.columns):
        if isinstance(regression[col][0], str):
            continue
        logger.debug(f"COL: {col}")
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_perfect_recovery():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (1, 0) with a chi squared of 0.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_perfect_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_sim.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--cheat_cc", "--no-sys_cov",
           "--prob_thresh", "0.5", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression_vals = [2.27e-5, 1.7, 0.0]
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        print("i: ", i, "col: ", col, "results[col]: ", results[col], "regression_vals[i]: ", regression_vals[i])
        np.testing.assert_allclose(results[col], regression_vals[i], rtol=1e-3, atol=1e-8)


def test_perfect_recovery_pz():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (1, 0) with a chi squared of 0. This test includes photo_zs.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_perfect_output_pz.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "--cheat_cc", "-o", str(outpath), "--no-sys_cov",
           "--prob_thresh", "0.5", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression_vals = [2.27e-5, 1.7, 0.0]
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], rtol=1e-3, atol=1e-8)


# Currently broken, pending fix
def test_calc_cov_term():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"
    runner.z_bins = np.arange(0, 1.4, 0.1)
    cov_mat = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.45, 0.5, 0.55], runner.z_bins, 1,
                                               survey)

    regression_cov = np.load(pathlib.Path(__file__).parent / "test_regression/test_cov_term.npy")

    plot = False
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(cov_mat, origin="lower")
        plt.colorbar()
        plt.title("Calculated Covariance Term")
        plt.subplot(1, 2, 2)
        plt.imshow(regression_cov, origin="lower")
        plt.colorbar()
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/check_test_cov_term.png")
    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


@pytest.mark.xfail(reason="This test is currently broken until new photoz runs")
def test_rescale_CC_for_cov():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "../config_des_data_zphot.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    _, _ = runner.unpack_dataframes()
    survey = "DES"
    runner.z_bins = np.arange(0.1, 1.0, 0.1)

    xx = np.array([0.05, 0.15865525, 0.5, 0.84134475, 0.95])  # 5 sigma points for normal dist
    X = norm(loc=1, scale=0.2)
    vals = X.ppf(xx)
    grid = np.meshgrid(vals, vals, vals, indexing="ij")
    grid0 = grid[0].flatten()
    grid1 = grid[1].flatten()
    grid2 = grid[2].flatten()

    seeds = np.array(np.arange(len(grid0)))
    rescale_vals = np.array([grid0, grid1, grid2, seeds]).T

    PROB_THRESH = 0.5

    cov_rate_norm = calculate_covariance_matrix_term(rescale_CC_for_cov, rescale_vals,
                                                     runner.z_bins, PROB_THRESH,
                                                     1, survey, runner.datasets,
                                                     runner.z_bins, False)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_regression/test_rescale_cov_term.npy")

    logger.debug(cov_rate_norm)
    plot = False
    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(cov_rate_norm, origin="lower")
        plt.colorbar()
        plt.title("Calculated Rescaled Covariance Term")
        plt.subplot(1, 2, 2)
        plt.imshow(regression_cov, origin="lower")
        plt.colorbar()
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_rescale_cov_term.png")
    np.testing.assert_allclose(cov_rate_norm, regression_cov, atol=1e-7)


def test_calc_effij():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config.yml"
    args.config = config_path
    args.cheat_cc = False
    args.debug = False
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    survey = "DES"
    # eff_ij = runner.calculate_transfer_matrix(survey, sim_z_col="SIM_ZCMB")
    # # Check that it is purely diagonal for this test case
    # print("EFF IJ: ", eff_ij)
    # for i in range(eff_ij.shape[0]):
    #     for j in range(eff_ij.shape[1]):
    #         if i != j:
    #             np.testing.assert_allclose(eff_ij[i, j], 0.0, atol=1e-7)

    eff_ij = runner.calculate_transfer_matrix(survey)
    # Check that it is purely diagonal for this test case
    eff_ij = eff_ij[1:-1, :]  # cutting off over and underflow bins in the true simulated redshift binning
    for i in range(eff_ij.shape[0]):
        for j in range(eff_ij.shape[1]):
            if i != j:
                np.testing.assert_allclose(eff_ij[i, j], 0.0, atol=1e-2)

    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_pz.yml"
    args.config = config_path
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    survey = "DES"
    eff_ij = runner.calculate_transfer_matrix(survey)
    eff_ij = eff_ij[1:-1, :]
    regression_eff_ij = np.load(pathlib.Path(__file__).parent / "test_regression/test_effij_regression.npy")
    np.testing.assert_allclose(eff_ij, regression_eff_ij, atol=1e-7)


def test_chi():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    args.debug = False
    runner = sauron_runner(args)
    datasets, surveys = runner.unpack_dataframes()
    runner.z_bins = np.arange(0, 1.4, 0.1)

    survey = "DES"
    index = 1
    N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(runner.z_bins)
    eff_ij = runner.calculate_transfer_matrix(survey)
    f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)) / \
        np.sum(datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins))
    n_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
    x = np.array([2.27e-5, 1.7])
    z_centers = 0.5 * (runner.z_bins[1:] + runner.z_bins[:-1])
    null_counts = calculate_null_counts(N_gen=N_gen, true_rate_function=power_law, rate_params=x, z_bins=runner.z_bins,
                                        z_centers=z_centers)

    regression_chi = 9.900071
    measured_chi = chi2(x, null_counts, f_norm, z_centers, eff_ij, n_data, power_law, x0 = [2.27e-5, 1.7])
    assert isinstance(measured_chi, float), "Measured chi is not a float."
    np.testing.assert_allclose(measured_chi, regression_chi, atol=1e-7)


@pytest.mark.xfail(reason="This test is currently broken until new photoz runs")
def test_regression_pz_5datasets_covariance():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Photo Zs.
       This also uses 5 datasets rather than 1 to test that functionality.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_regpz_sys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_5pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/test_regpz_sys_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-3 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression[col], atol=3e-4)
    # The tolerance here is much looser because the inclusion of systematics makes the results more stochastic.
    # The rescale CC for cov now uses deterministic preloaded values via inverse CDF, so the tolerance can be tightened.


def test_coverage_no_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (1, 0) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_coverage_nosys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_coverage_cc_fixed.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), '--no-sys_cov', "--prob_thresh", "0.5"]
    # Added --no-sys_cov flag here
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    logger.debug(f"Reading outpath {outpath}")
    df = pd.read_csv(outpath)

    sigma_1 = scipy_chi2.ppf([0.68], 2)
    sigma_2 = scipy_chi2.ppf([0.95], 2)

    product_2 = _coverage_chi2(df, truth=[2.27e-5, 1.7], param_names=("alpha", "beta"))


    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)
    plot = False
    if plot:
        import matplotlib.pyplot as plt

        plt.hist(product_2, bins=10, density=True, alpha=0.7, color="blue", label="Observed")
        x = np.linspace(0, 12, 100)
        # Dof = 6, 8 bins - 2 fitted parameters
        plt.plot(x, scipy_chi2.pdf(x, 2), color="red", linestyle="dashed", label="Expected")
        plt.axvline(sigma_1, color="r", linestyle="dashed", linewidth=1)
        plt.axvline(sigma_2, color="g", linestyle="dashed", linewidth=1)
        plt.xlabel("Chi-squared statistic")
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_coverage_nosys_hist.png")
        plt.clf()

        plt.errorbar(df["alpha"], df["beta"], xerr=df["alpha_error"], yerr=df["beta_error"], fmt="o", alpha=0.7)
        plt.scatter(2.27e-5, 1.7, color="red", label="Truth")
        plt.xlabel("Alpha")
        plt.ylabel("Beta")
        # # x = np.linspace(2.27e-5 - 5e-5, 2.27e-5 + 5e-5, 100)
        # # y = np.linspace(1.7 - 5, 1.7 + 5, 100)
        # # X, Y = np.meshgrid(x, y)
        # # X -= 2.27e-5
        # # Y -= 1.7
        # # pos = np.vstack([X.flatten(), Y.flatten()])
        # # mean_cov_inv = np.linalg.inv(mean_cov)

        # chi_product_1 = np.einsum('ij,jl->il', mean_cov_inv, pos)
        # chi_product_2 = np.einsum("il,il->l", pos, chi_product_1)

        x = np.linspace(5e-6, 4e-5, 100)
        y = np.linspace(0.8, 3.0, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        pos -= np.array([2.27e-5, 1.7])

        chivals = np.einsum("...i,ij,...j->...", pos, np.linalg.inv(mean_cov), pos)

        plt.contour(X, Y, chivals, levels=[2.3, 6], colors=["red", "black"], linestyles=["dashed", "dashed"])
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_coverage_nosys_alpha_beta.png")

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")
    logger.debug(f"df alpha: {df['alpha']}")

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.1)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.1)

    # We also perform some additional strict testing with these tolerances.

    # Finally we also check using a KS test that the observed distribution is consistent with chi2 with 2 dofs.
    np.random.seed(seed=42)
    simulated_data = scipy_chi2.rvs(df=2, size=50, scale=1.0)
    p_value = ks_2samp(simulated_data, product_2)
    np.testing.assert_array_less(0.05, p_value.pvalue)


def test_coverage_with_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (2.27e-5, 1.7) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_coverage_sys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_coverage_cc_fixed.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    df = pd.read_csv(outpath)

    sigma_1 = scipy_chi2.ppf([0.68], 2)
    sigma_2 = scipy_chi2.ppf([0.95], 2)

    product_2 = _coverage_chi2(df, truth=[2.27e-5, 1.7], param_names=("alpha", "beta"))

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    plot = False
    if plot:
        import matplotlib.pyplot as plt

        plt.hist(product_2, bins=10, density=True, alpha=0.7, color="blue", label="Observed")
        x = np.linspace(0, 12, 100)
        # Dof = 6, 8 bins - 2 fitted parameters
        plt.plot(x, scipy_chi2.pdf(x, 2), color="red", linestyle="dashed", label="Expected")
        plt.axvline(sigma_1, color="r", linestyle="dashed", linewidth=1)
        plt.axvline(sigma_2, color="g", linestyle="dashed", linewidth=1)
        plt.xlabel("Chi-squared statistic")
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_coverage_sys_hist.png")

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    # The expected coverages are the nominal Gaussian 1σ and 2σ fractions (≈0.68 and ≈0.95), but in this
    # test we only have O(50) pseudo-experiments (len(product_2)). The realised fractions therefore have
    # binomial sampling noise of order sqrt(p * (1 - p) / N) ≈ 0.07 for p ≈ 0.68 and N ≈ 50. We then round
    # to the nearest whole number of tests (4/50) for a cut of 0.08. In addition,
    # the test statistic is chi-squared–like rather than exactly Gaussian, which further broadens the
    # empirical distribution. We therefore use atol=0.08 to avoid flaky failures while still detecting
    # substantial coverage regressions; tighter tolerances (e.g. 0.05) were observed to fail spuriously.
    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.08)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.08)

    # Finally we also check using a KS test that the observed distribution is consistent with chi2 with 2 dofs.
    np.random.seed(seed=42)
    simulated_data = scipy_chi2.rvs(df = 2, size=50, scale=1.0)
    p_value = ks_2samp(simulated_data, product_2)
    np.testing.assert_array_less(0.05, p_value.pvalue)


def test_perfect_recovery_multisurvey():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (2.27e-5, 1.7) with a chi squared of 0.
    This time, we do DES, LOWZ and ROMAN together.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_perfect_output_multisurvey.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_sim_multisurvey.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--cheat_cc", "--no-sys_cov", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression_vals = [2.27e-5, 1.7, 0.0]
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-6)  # atol not rtol b/c we expect 0


def test_regression_multisurvey():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    This time, we do DES, LOWZ and ROMAN together.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_regmultisurvey_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_multisurvey.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov", "--no-sanity-check"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/test_regmultisurvey_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_apply_cut():
    """Test that apply_cuts method in sauron.py works as expected."""
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_cuts.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    survey = "DES"
    n_before = len(runner.datasets[f"{survey}_DATA_ALL_1"].df)

    max_x1 = np.max(runner.datasets[f"{survey}_DATA_ALL_1"].df["x1"])
    assert max_x1 > 2, "Dataset is already within cut range pre cut."
    min_x1 = np.min(runner.datasets[f"{survey}_DATA_ALL_1"].df["x1"])
    assert min_x1 < -2, "Dataset is already within cut range pre cut."
    max_c = np.max(runner.datasets[f"{survey}_DATA_ALL_1"].df["c"])
    assert max_c >= 0.3, "Dataset is already within cut range pre cut."
    min_c = np.min(runner.datasets[f"{survey}_DATA_ALL_1"].df["c"])
    assert min_c <= -0.3, "Dataset is already within cut range pre cut."

    runner.apply_cuts(survey)
    n_after = len(runner.datasets[f"{survey}_DATA_ALL_1"].df)

    assert n_after < n_before, "apply_cuts did not reduce the number of SNe as expected."

    max_x1 = np.max(runner.datasets[f"{survey}_DATA_ALL_1"].df["x1"])
    assert max_x1 <= 2, "apply_cuts did not apply the x1 cut correctly."
    min_x1 = np.min(runner.datasets[f"{survey}_DATA_ALL_1"].df["x1"])
    assert min_x1 >= -2, "apply_cuts did not apply the x1 cut correctly."
    max_c = np.max(runner.datasets[f"{survey}_DATA_ALL_1"].df["c"])
    assert max_c <= 0.3, "apply_cuts did not apply the cut correctly."
    min_c = np.min(runner.datasets[f"{survey}_DATA_ALL_1"].df["c"])
    assert min_c >= -0.3, "apply_cuts did not apply the cut correctly."

    survey = "DES_BAD_1"
    with pytest.raises(ValueError, match="Invalid cut specification"):
        runner.apply_cuts(survey)


def test_des_data_regression():
    """Regression test on real DES data with photo-z to ensure Sauron output is unchanged."""
    outpath = pathlib.Path(__file__).parent / "test_output/test_desdatareg_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "./test_configs/config_des_data_test.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/test_desdatareg_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_cc_decontam():
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_coverage_cc_fixed.yml"
    args = SimpleNamespace()
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.z_bins = np.linspace(0.1, 1.0, 11)
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"

    PROB_THRESH = 0.5

    pulls = []

    pulls = np.empty((50, len(runner.z_bins)-1))
    all_ntrue = np.empty((50, len(runner.z_bins)-1))
    all_ncalc = np.empty((50, len(runner.z_bins)-1))
    for i in range(50):
        index = i+1
        logger.debug(f"Working on survey {survey}, dataset {index} -------------------")
        runner.fit_args_dict["z_bins"][survey] = runner.z_bins
        n_calc = runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=False)

        n_true = runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
        residual = n_true - n_calc
        pull = residual / np.sqrt(n_true)
        logger.debug(f"n_true: {n_true}")
        logger.debug(f"n_calc: {n_calc}")

        pulls[i, :] = pull
        all_ntrue[i, :] = n_true
        all_ncalc[i, :] = n_calc

    pulls = np.array(pulls)

    means = np.mean(pulls, axis=0)

    logger.debug(f"MEANS: {means}")

    mean_ntrue = np.mean(all_ntrue, axis=0)
    mean_ncalc = np.mean(all_ncalc, axis=0)
    mean_res = np.mean(all_ntrue - all_ncalc, axis=0)
    std_ntrue = np.std(all_ntrue, axis=0)
    std_ncalc = np.std(all_ncalc, axis=0)

    z_centers = (runner.z_bins[:-1] + runner.z_bins[1:]) / 2
    plot = False
    if plot:
        plt.clf()

        # Print the following in a form that can be copied into a python list
        print("z_centers = ", list(z_centers))
        print("mean_res = ", list(mean_res))
        print("std_ntrue = ", list(std_ntrue))

        plt.errorbar(z_centers, mean_res, yerr=std_ntrue/np.sqrt(50), fmt='o', label='True - Calculated CC Counts')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Redshift')
        plt.ylabel('CC Counts')
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_cc_decontam_counts.png")

    np.testing.assert_allclose(means, 0.0, atol=1/np.sqrt(50))


def test_cc_decontam_small():
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_5pz.yml"
    args = SimpleNamespace()
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.z_bins = np.linspace(0.1, 1.0, 8)
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"
    runner.apply_cuts(survey)

    PROB_THRESH = 0.5
    logger.debug("successfully reran")

    pulls = []
    n_trials = 5

    pulls = np.empty((n_trials, len(runner.z_bins)-1))
    all_ntrue = np.empty((n_trials, len(runner.z_bins)-1))
    all_ncalc = np.empty((n_trials, len(runner.z_bins)-1))
    all_ncc = np.empty((n_trials, len(runner.z_bins)-1))
    for i in range(n_trials):
        index = i+1
        logger.debug(f"Working on survey {survey}, dataset {index} -------------------")
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.plot(runner.datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins), label="Sim IA Counts")
        plt.plot(runner.datasets[f"{survey}_SIM_CC"].z_counts(runner.z_bins), label="Sim CC Counts")
        plt.plot(runner.datasets[f"{survey}_SIM_ALL"].z_counts(runner.z_bins), label="Sim All Counts")
        plt.plot(runner.datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins, prob_thresh=PROB_THRESH), ls="--",
                 label="Sim IA Counts Cut")
        plt.plot(runner.datasets[f"{survey}_SIM_CC"].z_counts(runner.z_bins, prob_thresh=PROB_THRESH), ls="--" ,
                 label="Sim CC Counts Cut")
        plt.plot(runner.datasets[f"{survey}_SIM_ALL"].z_counts(runner.z_bins, prob_thresh=PROB_THRESH), ls="--",
                 label="Sim All Counts Cut")
        plt.yscale("log")
        plt.legend()

        bias_cor = runner.datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins) / runner.datasets[f"{survey}_SIM_ALL"].z_counts(runner.z_bins, prob_thresh = 0.5)
        plt.subplot(1, 2, 2)
        plt.plot(runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins), label="Data IA Counts")
        plt.plot(runner.datasets[f"{survey}_DATA_CC_{index}"].z_counts(runner.z_bins), label="Data CC Counts")
        plt.plot(runner.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(runner.z_bins), label="Data All Counts")
        plt.plot(runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins, prob_thresh = PROB_THRESH), ls = "--", label="Data IA Counts Cut")
        plt.plot(runner.datasets[f"{survey}_DATA_CC_{index}"].z_counts(runner.z_bins, prob_thresh = PROB_THRESH), ls = "--", label="Data CC Counts Cut")
        plt.plot(runner.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(runner.z_bins, prob_thresh = PROB_THRESH)*bias_cor, color = "k", lw= 3, ls = "--", label="Data All Counts Cut w/ BCor")
        plt.yscale("log")
        plt.legend()
        plt.savefig(pathlib.Path(__file__).parent / f"test_plots/aaaa_test_cc_decontam_simcounts_{index}_small.png")

        runner.fit_args_dict["z_bins"][survey] = runner.z_bins
        n_calc = runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=False)

        n_true = runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
        n_CC = runner.datasets[f"{survey}_DATA_CC_{index}"].z_counts(runner.z_bins)
        residual = n_true - n_calc
        pull = residual / np.sqrt(n_true)

        pulls[i, :] = pull
        all_ntrue[i, :] = n_true
        all_ncalc[i, :] = n_calc
        all_ncc[i, :] = n_CC
        # pulls.extend(list(pull))

    pulls = np.array(pulls)

    means = np.mean(pulls, axis=0)
    stds = np.std(pulls, axis=0)

    logger.debug(f"MEANS: {means}")

    mean_ntrue = np.mean(all_ntrue, axis=0)
    mean_ncalc = np.mean(all_ncalc, axis=0)
    std_ntrue = np.std(all_ntrue, axis=0)
    std_ncalc = np.std(all_ncalc, axis=0)

    z_centers = (runner.z_bins[:-1] + runner.z_bins[1:]) / 2
    plt.clf()
    plt.errorbar(z_centers, mean_ntrue, yerr=std_ntrue, fmt="o", label="True CC Counts")
    plt.errorbar(z_centers, mean_ncalc, yerr=std_ncalc, fmt="o", label="Calculated CC Counts")
    plt.xlabel("Redshift")
    plt.ylabel("CC Counts")
    plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_cc_decontam_counts_small.png")

    np.testing.assert_allclose(means, 0.0, atol=1/np.sqrt(n_trials))


def test_regression_SDSS():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    This time, we do DES and SDSS together.
    """
    outpath = pathlib.Path(__file__).parent / "test_SDSS_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_DES_SDSS_master.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/SDSS_multi_regression.csv")
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_coverage_SDSS():
    """In this test we check the coverage properties of SAURON on the 50 SDSS sim datasets.
        We should recover the truth (2.27e-5, 1.7) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_coverage_SDSS_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/config_SDSS_coverage.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    df = pd.read_csv(outpath)

    sigma_1 = scipy_chi2.ppf([0.68], 2)
    sigma_2 = scipy_chi2.ppf([0.95], 2)

    product_2 = _coverage_chi2(df, truth=[2.27e-5, 1.7], param_names=("alpha", "beta"))

    for i in range(len(product_2)):
        logger.debug(f"Product 2 for dataset {i}: {product_2[i]}")

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    plot = True
    if plot:
        # import matplotlib.pyplot as plt

        # plt.hist(product_2, bins=10, density=True, alpha=0.7, color='blue', label='Observed')
        # x = np.linspace(0, 12, 100)
        # # Dof = 6, 8 bins - 2 fitted parameters
        # plt.plot(x, scipy_chi2.pdf(x, 2), color='red', linestyle='dashed', label='Expected')
        # plt.axvline(sigma_1, color='r', linestyle='dashed', linewidth=1)
        # plt.axvline(sigma_2, color='g', linestyle='dashed', linewidth=1)
        # plt.xlabel("Chi-squared statistic")
        # plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_coverage_sys_hist_SDSS.png")
        outpath = pathlib.Path(__file__).parent / "test_plots/test_coverage_sys_scatter_SDSS.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        sauron_coverage_scatterplot(df, param_1_name="alpha", param_2_name="beta", save=True, outpath=outpath)

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    # The expected coverages are the nominal Gaussian 1σ and 2σ fractions (≈0.68 and ≈0.95), but in this
    # test we only have O(50) pseudo-experiments (len(product_2)). The realised fractions therefore have
    # binomial sampling noise of order sqrt(p * (1 - p) / N) ≈ 0.07 for p ≈ 0.68 and N ≈ 50. We then round
    # to the nearest whole number of tests (4/50) for a cut of 0.08. In addition,
    # the test statistic is chi-squared–like rather than exactly Gaussian, which further broadens the
    # empirical distribution. We therefore use atol=0.08 to avoid flaky failures while still detecting
    # substantial coverage regressions; tighter tolerances (e.g. 0.05) were observed to fail spuriously.
    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.08)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.08)

    # Finally we also check using a KS test that the observed distribution is consistent with chi2 with 2 dofs.
    np.random.seed(seed=42)
    simulated_data = scipy_chi2.rvs(df = 2, size=50, scale=1.0)
    p_value = ks_2samp(simulated_data, product_2)
    np.testing.assert_array_less(0.05, p_value.pvalue)



def test_coverage_SDSS_plus_DES():
    """In this test we check the coverage properties of SAURON on the 50 SDSS sim datasets.
        We should recover the truth (2.27e-5, 1.7) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_coverage_SDSS_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_SDSS_DES_coverage.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    _run_cmd(cmd)
    df = pd.read_csv(outpath)

    df = df[df["survey"].str.contains("combined")]

    sigma_1 = scipy_chi2.ppf([0.68], 2)
    sigma_2 = scipy_chi2.ppf([0.95], 2)

    product_2 = _coverage_chi2(df, truth=[2.27e-5, 1.7], param_names=("alpha", "beta"))

    for i in range(len(product_2)):
        logger.debug(f"Product 2 for dataset {i}: {product_2[i]}")

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    plot = True
    if plot:
        import matplotlib.pyplot as plt

        plt.hist(product_2, bins=10, density=True, alpha=0.7, color='blue', label='Observed')
        x = np.linspace(0, 12, 100)
        # Dof = 6, 8 bins - 2 fitted parameters
        plt.plot(x, scipy_chi2.pdf(x, 2), color='red', linestyle='dashed', label='Expected')
        plt.axvline(sigma_1, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(sigma_2, color='g', linestyle='dashed', linewidth=1)
        plt.xlabel("Chi-squared statistic")
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_coverage_sys_hist_SDSS_plus_DES.png")

        sauron_coverage_scatterplot(df, outpath=pathlib.Path(__file__).parent / "test_plots/test_coverage_sys_scatter_SDSS_plus_DES.png")



    # The expected coverages are the nominal Gaussian 1σ and 2σ fractions (≈0.68 and ≈0.95), but in this
    # test we only have O(50) pseudo-experiments (len(product_2)). The realised fractions therefore have
    # binomial sampling noise of order sqrt(p * (1 - p) / N) ≈ 0.07 for p ≈ 0.68 and N ≈ 50. We then round
    # to the nearest whole number of tests (4/50) for a cut of 0.08. In addition,
    # the test statistic is chi-squared–like rather than exactly Gaussian, which further broadens the
    # empirical distribution. We therefore use atol=0.08 to avoid flaky failures while still detecting
    # substantial coverage regressions; tighter tolerances (e.g. 0.05) were observed to fail spuriously.
    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.08)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.08)

    # Finally we also check using a KS test that the observed distribution is consistent with chi2 with 2 dofs.
    np.random.seed(seed=42)
    simulated_data = scipy_chi2.rvs(df = 2, size=50, scale=1.0)
    p_value = ks_2samp(simulated_data, product_2)
    np.testing.assert_array_less(0.05, p_value.pvalue)



def test_cc_decontam_SDSS():
    config_path = pathlib.Path(__file__).parent / "test_configs/config_SDSS_coverage.yml"
    args = SimpleNamespace()
    args.config = config_path
    args.cheat_cc = True
    runner = sauron_runner(args)
    runner.z_bins = np.linspace(0.1, 0.4, 8)
    datasets, surveys = runner.unpack_dataframes()
    survey = "SDSS"

    PROB_THRESH = 0.5

    pulls = []

    pulls = np.empty((50, len(runner.z_bins)-1))
    all_ntrue = np.empty((50, len(runner.z_bins)-1))
    all_ncalc = np.empty((50, len(runner.z_bins)-1))
    for i in range(50):
        index = i+1
        logger.debug(f"Working on survey {survey}, dataset {index} -------------------")
        runner.fit_args_dict['z_bins'][survey] = runner.z_bins
        args.cheat_cc = False
        n_calc = runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=False)

        n_true = runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
        residual = n_true - n_calc
        pull = residual / np.sqrt(n_true)

        pulls[i, :] = pull
        all_ntrue[i, :] = n_true
        all_ncalc[i, :] = n_calc

    pulls = np.array(pulls)


    means = np.mean(pulls, axis=0)

    logger.debug(f"MEANS: {means}")

    mean_ntrue = np.mean(all_ntrue, axis=0)
    mean_ncalc = np.mean(all_ncalc, axis=0)
    mean_res = np.mean(all_ntrue - all_ncalc, axis=0)
    std_ntrue = np.std(all_ntrue, axis=0)
    std_ncalc = np.std(all_ncalc, axis=0)

    z_centers = (runner.z_bins[:-1] + runner.z_bins[1:]) / 2
    plot = False
    if plot:
        plt.clf()

        # Print the following in a form that can be copied into a python list
        logging.debug("z_centers = ", list(z_centers))
        logging.debug("mean_res = ", list(mean_res))
        logging.debug("std_ntrue = ", list(std_ntrue))

        plt.errorbar(z_centers, mean_res, yerr=std_ntrue/np.sqrt(50), fmt='o', label='True - Calculated CC Counts')
        plt.axhline(0, color='k', linestyle='--')
        plt.xlabel('Redshift')
        plt.ylabel('CC Counts')
        plt.savefig(pathlib.Path(__file__).parent / "test_plots/test_cc_decontam_counts_SDSS.png")


    np.testing.assert_allclose(means, 0.0, atol=1/np.sqrt(50))



import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2 as scipy_chi2
import numpy as np
from funcs import mean_of_correlated_errors

from matplotlib.patches import Ellipse

def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, facecolor='none', **kwargs):
    """
    Plots a covariance error ellipse given a 2x2 covariance matrix and a center.
    """
    # 1. Calculate eigenvalues and eigenvectors
    # eigh is optimized for symmetric matrices like covariance matrices
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    print("Eigenvalues before sorting:", eigenvalues)

    # Sort eigenvalues in descending order to identify major vs minor axes
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 2. Calculate the angle of rotation (in degrees)
    # The angle is determined by the first component of the primary eigenvector
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    print("Angle of rotation (degrees):", angle)

    # 3. Calculate width and height based on the standard deviation scale
    # The lengths of the axes are proportional to the square root of the eigenvalues
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # 4. Construct and add the Ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      facecolor=facecolor, **kwargs)

    ax.add_patch(ellipse)
    return ellipse

def sauron_coverage_scatterplot(results, param_1_name = "alpha", param_2_name = "beta",
save = True, outpath = pathlib.Path(__file__).parent / "test_plots/coverage_scatter.png"):


    plt.figure(figsize = (8,4), dpi=200)
    plt.subplot(1,2,1)


    a = param_1_name
    b = param_2_name


    list_of_xj = [np.array([results[a].iloc[i], results[b].iloc[i]]) for i in range(len(results))]
    list_of_Cj = [np.array([[results[f"{a}_error"].iloc[i]**2, results[f"cov_{a}_{b}"].iloc[i]],
                            [results[f"cov_{a}_{b}"].iloc[i], results[f"{b}_error"].iloc[i]**2]]) for i in range(len(results))]
    weighted_average, mean_cov = mean_of_correlated_errors(list_of_xj, list_of_Cj)



    weighted_alpha_mean = weighted_average[0]
    weighted_beta_mean = weighted_average[1]
    #alpha_err = np.sqrt(mean_cov[0,0])
    #beta_err = np.sqrt(mean_cov[1,1])
    #average_covariance = mean_cov[0,1]
    #average_covariance_matrix = mean_cov



    # weighted_alpha_mean = np.sum(results[a] / results[f"{a}_error"]**2) / np.sum(1 / results[f"{a}_error"]**2)
    # weighted_beta_mean = np.sum(results[b] / results[f"{b}_error"]**2) / np.sum(1 / results[f"{b}_error"]**2)
    alpha_err = np.sqrt(np.sum(results[f"{a}_error"]**2)) / len(results)
    beta_err = np.sqrt(np.sum(results[f"{b}_error"]**2)) / len(results)
    average_covariance = np.mean(results[f"cov_{a}_{b}"])
    # print("Average Covariance:", average_covariance)
    # print("reduced covariance:", np.mean(results[f"cov_{a}_{b}"] / (results[f"{a}_error"] * results[f"{b}_error"])))
    #plt.errorbar(np.mean(results[a]),np.mean(results[b]), xerr = alpha_err, yerr = beta_err, fmt = "o", color = "k", label = "Mean", ms = 5)

    print("Weighted Alpha Mean:", weighted_alpha_mean)
    print("Weighted Beta Mean:", weighted_beta_mean)
    plt.errorbar(weighted_alpha_mean, weighted_beta_mean, xerr = alpha_err, yerr = beta_err, fmt = "o", color = "k", label = "Weighted Mean", ms = 5)
    print("Unweighted Alpha Mean:", np.mean(results[a]))
    print("Unweighted Beta Mean:", np.mean(results[b]))

    average_beta_err = np.average(results[f"{b}_error"], weights = 1 / results[f"{b}_error"]**2)
    average_alpha_err = np.average(results[f"{a}_error"], weights = 1 / results[f"{a}_error"]**2)
    print("Average Alpha Error:", average_alpha_err)
    print("Average Beta Error:", average_beta_err)

    average_covariance_matrix = np.array([[alpha_err**2, average_covariance],
                                        [average_covariance, beta_err**2]])

    average_covariance_matrix = np.array([[np.mean(results[f"{a}_error"]**2), average_covariance],
                                        [average_covariance, np.mean(results[f"{b}_error"]**2)]])

    total_inv_cov = np.zeros((2,2))
    # for i in range(len(results)):
    #     print(f"############ {i} ############")
    #     print("alpha_err:", results[f"{a}_error"][i])
    #     print("beta_err:", results[f"{b}_error"][i])
    #     print("covariance:", results[f"cov_{a}_{b}"][i])
    #     total_inv_cov += np.linalg.inv(np.array([[results[f"{a}_error"][i]**2, results[f"cov_{a}_{b}"][i]],
    #                                             [results[f"cov_{a}_{b}"][i], results[f"{b}_error"][i]**2]]))
    #     total_cov = np.linalg.inv(total_inv_cov)
    #     print("Total Covariance Matrix:", total_cov)

    #     alpha_mean = np.mean(results[a])
    #     beta_mean = np.mean(results[b])
    #     print("Mahalnobis distance of mean to truth using total covariance")
    #     distance = np.array([alpha_mean, beta_mean]) - np.array([2.27e-5, 1.7])
    #     maha_dist = distance.T @ np.linalg.inv(total_cov) @ distance
    #     p_value = 1 - scipy_chi2.cdf(maha_dist, df=2)
    #     print("Chi2 cdf:", scipy_chi2.cdf(maha_dist, df=2))
    #     print("p_value:", p_value)
    #     print("Mahalanobis Distance to Simulated Alpha & Beta:", maha_dist)

    dist = np.array([weighted_alpha_mean, weighted_beta_mean]) - np.array([2.27e-5, 1.7])
    maha_dist = dist.T @ np.linalg.inv(average_covariance_matrix) @ dist
    from scipy.stats import chi2
    p_value = 1 - scipy_chi2.cdf(maha_dist, df=2)
    print("Chi2 cdf:", scipy_chi2.cdf(maha_dist, df=2))
    print("p_value:", p_value)
    print("Mahalanobis Distance to Simulated Alpha & Beta:", maha_dist)

    plt.axvline(2.27e-5, color = "k", ls = "--", label = "Simulated Rate")
    plt.axhline(1.7, color = "k", ls = "--")


    x = np.linspace(5e-6, 4e-5, 100)
    y = np.linspace(0.8, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    pos -= np.array([2.27e-5, 1.7])
    from scipy.stats import multivariate_normal


    chivals = np.einsum('...i,ij,...j->...', pos, np.linalg.inv(average_covariance_matrix), pos)
    all_dist = np.array(results[[f"{a}", f"{b}"]]) - np.array([2.27e-5, 1.7])

    data_chivals = np.einsum('...i,ij,...j->...', all_dist, np.linalg.inv(average_covariance_matrix), all_dist)

    integer_sigma = np.empty_like(data_chivals)
    integer_sigma[data_chivals < 2.30] = 1
    integer_sigma[(data_chivals >= 2.30) & (data_chivals < 6.18)] = 2
    integer_sigma[data_chivals >= 6.18] = 3


    print("1 sigma points:", np.sum(integer_sigma == 1))
    print("2 sigma points:", np.sum(integer_sigma == 2))
    print("3 sigma points:", np.sum(integer_sigma == 3))

    labels = [r"< 1 $\sigma$", r"1-2 $\sigma$", r"> 2 $\sigma$"]
    for sig in [1, 2, 3]:
        plt.errorbar(results[f"{a}"][integer_sigma == sig], results[f"{b}"][integer_sigma == sig],
        xerr = results[f"{a}_error"][integer_sigma == sig], yerr = results[f"{b}_error"][integer_sigma == sig],
        fmt = "o", label = labels[sig-1], zorder = 0, ms = 3, alpha = 0.3)

    #chivals = pos.T @ np.linalg.inv(average_covariance_matrix) @ pos
    plt.contour(X, Y, chivals, levels=[2.30, 6.18], colors=['blue', 'red'], linestyles=['--', '--'], label = "1 and 2 sigma Contours")

    #from scipy.stats import multivariate_normal

    list_of_xj = [np.array([results[a].iloc[i], results[b].iloc[i]]) for i in range(len(results))]
    list_of_Cj = [np.array([[results[f"{a}_error"].iloc[i]**2, results[f"cov_{a}_{b}"].iloc[i]],
                            [results[f"cov_{a}_{b}"].iloc[i], results[f"{b}_error"].iloc[i]**2]]) for i in range(len(results))]
    weighted_average, mean_cov = mean_of_correlated_errors(list_of_xj, list_of_Cj)
    plot_covariance_ellipse(ax=plt.gca(), mean=weighted_average, cov=mean_cov, n_std=1, edgecolor='red')
    plot_covariance_ellipse(ax=plt.gca(), mean=weighted_average, cov=mean_cov, n_std=2, edgecolor='blue')
    plot_covariance_ellipse(ax=plt.gca(), mean=weighted_average, cov=mean_cov, n_std=3, edgecolor='green')

    #plt.xlim(np.min(results[f"{a}"])*0.9, np.max(results[f"{a}"])*1.1)
    #plt.ylim(np.min(results[f"{b}"])*0.9, np.max(results[f"{b}"])*1.1)

    plt.xlim(2.1e-5, 2.3e-5)
    plt.ylim(1.6, 1.8)

    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\beta$")

    plt.legend()
    ##############
    print("############################")


    plt.subplot(1,2,2)

    df = results

    sigma_1 = scipy_chi2.ppf([0.68], 2)
    sigma_2 = scipy_chi2.ppf([0.95], 2)

    a = np.median(df["alpha_error"]**2)
    b = np.median(df["beta_error"]**2)
    c = np.median(df["cov_alpha_beta"])

    mean_cov = np.array([[a, c], [c, b]])

    all_alpha = df["alpha"] - 2.27e-5
    all_beta = df["beta"] - 1.7
    inv_cov = np.linalg.inv(mean_cov)
    all_pos = np.vstack([all_alpha, all_beta])
    product_1 = np.einsum('ij,jl->il', inv_cov, all_pos)
    product_2 = np.einsum("il,il->l", all_pos, product_1)

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)


    plt.hist(product_2, bins=10, density=True, alpha=0.7, color='blue', label='Observed')
    x = np.linspace(0, 14, 100)
    # Dof = 6, 8 bins - 2 fitted parameters
    plt.plot(x, scipy_chi2.pdf(x, 2), color='red', linestyle='dashed', label='Expected')
    plt.axvline(sigma_1, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(sigma_2, color='g', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("$\\chi^2$ statistic")
    plt.ylabel("Density")
    plt.ylim(0, 0.4)

    plt.subplots_adjust(wspace=0.35)

    plt.suptitle("Coverage Test for Rate Recovery", y=0.95)

    if save:
        plt.savefig(outpath)
    else:
        plt.show()
def test_regression_power_law_DTD():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    This time, we do DES and SDSS together.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_DTD_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_DES_SDSS_DTD.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/DES_SDSS_DTD_regression.csv")
    for i, col in enumerate([r"beta", r"R_1", r"beta_error", r"R_1_error", r"cov_beta_R_1", "reduced_chi_squared"]):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


# I have disabled this test for two reasons. 1 it is too volatile with the small redshift data
# Secondly, the paper does not calculate the binned DTD way, rather fitting to the binned rate. I should
# make a test that does that.

# def test_regression_binned_DTD():
#     """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
#     This time, we do DES and SDSS together.
#     """
#     outpath = pathlib.Path(__file__).parent / "test_output/test_binned_DTD_output.csv"
#     if os.path.exists(outpath):
#         os.remove(outpath)
#     sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
#     config_path = pathlib.Path(__file__).parent / "test_configs/test_config_DES_SDSS_DTD_binned.yml"
#     cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
#     result = subprocess.run(cmd, capture_output=False, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"Command failed with exit code {result.returncode}\n"
#             f"stdout:\n{result.stdout}\n"
#             f"stderr:\n{result.stderr}"
#         )

#     results = pd.read_csv(outpath)
#     regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/DES_SDSS_binned_DTD_regression.csv")
#     for i, col in enumerate([r"param_0", r"param_1", r"param_2", r"param_0_error", r"param_1_error", r"param_2_error", r"cov_param_0_param_1", r"cov_param_0_param_2", r"cov_param_1_param_2", "reduced_chi_squared"]):
#         try:
#             np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
#         except AssertionError as e:
#             logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
#                            f"Please check if this is expected. ")
#             logger.warning(str(e))
#         np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_regression_CSFR_list():
    """In this test, we check the CSFR list functionality. This functions as a regression test, AND, the results for the
    B13 CSFR should be equivalent to the power law CSFR test, as that one uses B13.
    """
    outpath = pathlib.Path(__file__).parent / "test_output/test_CSFR_list_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    assert not os.path.exists(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_DES_SDSS_DTD_CSFR_list.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]


    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)

    # Ensure the multi-CSFR loop actually ran
    assert {"B13_uncorrected", "S20"}.issubset(set(results["csfr"].unique()))

    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/DES_SDSS_CSFR_list_regression.csv").reset_index(drop=True)
    for i, col in enumerate([r"beta", r"R_1", r"beta_error", r"R_1_error", r"cov_beta_R_1", "reduced_chi_squared"]):
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)
    # Now check it matches the other test
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regression/DES_SDSS_DTD_regression.csv")
    results = results.loc[results["csfr"] == "B13_uncorrected"]
    for i, col in enumerate([r"beta", r"R_1", r"beta_error", r"R_1_error", r"cov_beta_R_1", "reduced_chi_squared"]):
        logger.debug(f"Checking {col}")
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


def test_regression_AplusB():
    """In this test, we check the AplusB DTD regresses."""
    outpath = pathlib.Path(__file__).parent / "test_output/test_AplusB_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"

    config_path = pathlib.Path(__file__).parent / "test_configs/test_config_DES_SDSS_AplusB_DTD.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath).sort_values(["survey", "csfr"]).reset_index(drop=True)

    # Ensure the multi-CSFR loop actually ran
    assert {"B13", "S20"}.issubset(set(results["csfr"].unique()))

    regression = (
        pd.read_csv(pathlib.Path(__file__).parent / "test_regression/DES_SDSS_AplusB_DTD_regression.csv")
        .sort_values(["survey", "csfr"])
        .reset_index(drop=True)
    )

    A_resids = results["A"] - regression["A"]
    B_resids = results["B"] - regression["B"]
    A_pulls = A_resids / results["A_error"]
    B_pulls = B_resids / results["B_error"]
    print("A pulls:", A_pulls)
    print("B pulls:", B_pulls)

    for i, col in enumerate([r"A", r"B", r"A_error", r"B_error", r"cov_A_B", "reduced_chi_squared"]):
        logger.debug(f"Checking {col}")
        try:
            np.testing.assert_allclose(results[col], regression[col], rtol=warning_rtol)
        except AssertionError as e:
            logger.warning(f"Values for {col} have changed more than the warning tolerance of {warning_rtol}. "
                           f"Please check if this is expected. ")
            logger.warning(str(e))
        np.testing.assert_allclose(results[col], regression[col], rtol=global_rtol)


# ----------------------------------------------------------------------------------------------------
# FITOPT parsing (SYSTEMATIC_FITOPTS / FIDUCIAL_FITOPT)
# ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    (0, [0]),
    ("001", [1]),
    ("1,3", [1, 3]),
    ("3, 1,", [1, 3]),
    ([3, 1, 1], [1, 3]),
    (["2", 0], [0, 2]),
])
def test_parse_fitopt_list(value, expected):
    """FIDUCIAL_FITOPT / SYSTEMATIC_FITOPTS values become sorted lists of unique ints."""
    from runner import _parse_fitopt_list
    assert _parse_fitopt_list(value, "FIDUCIAL_FITOPT", "DES") == expected


@pytest.mark.parametrize("value", ["a", -1, [], "", "1.5", True])
def test_parse_fitopt_list_bad(value):
    from runner import _parse_fitopt_list
    with pytest.raises(ValueError):
        _parse_fitopt_list(value, "SYSTEMATIC_FITOPTS", "DES")


def _make_fitopt_files(root, name, fitopts, n_realizations=1):
    """Write empty FITOPTnnn.FITRES.gz files for one SIM/DATA entry; return the FITOPT000 path(s)."""
    paths = []
    for i in range(n_realizations):
        d = root / f"{name}-{i + 1:04d}"
        d.mkdir(parents=True, exist_ok=True)
        for fitopt in fitopts:
            (d / f"FITOPT{fitopt:03d}.FITRES.gz").touch()
        paths.append(str(d / "FITOPT000.FITRES.gz"))
    return paths


def _fitopt_survey_dict(paths, systematic=None, fiducial=None):
    survey_dict = {"FIT_OPTIONS": {"RATE_FUNC": "power_law"}}
    if systematic is not None:
        survey_dict["FIT_OPTIONS"]["SYSTEMATIC_FITOPTS"] = systematic
    if fiducial is not None:
        survey_dict["FIT_OPTIONS"]["FIDUCIAL_FITOPT"] = fiducial
    survey_dict["DUMP_IA"] = {"PATH": "/does/not/matter/DUMP.gz"}  # DUMP files have no fitopts
    for key, p in paths.items():
        survey_dict[key] = {"PATH": p if len(p) > 1 else p[0], "ZCOL": "zHD"}
    return survey_dict


def test_parse_fitopt_options(tmp_path):
    """parse_fitopt_options resolves ALL / NONE / explicit fitopts and checks the sim and data files exist."""
    import yaml
    from runner import sauron_runner

    fitopts = [0, 1, 2, 3]
    sep = {
        "SIM_IA": _make_fitopt_files(tmp_path, "SIM_IA", fitopts),
        "SIM_CC": _make_fitopt_files(tmp_path, "SIM_CC", fitopts),
        "DATA_IA": _make_fitopt_files(tmp_path, "DATA_IA", fitopts, n_realizations=2),
        "DATA_CC": _make_fitopt_files(tmp_path, "DATA_CC", fitopts, n_realizations=2),
    }
    combined = {
        "SIM_ALL": _make_fitopt_files(tmp_path, "SIM_ALL", fitopts),
        "DATA_ALL": _make_fitopt_files(tmp_path, "DATA_ALL", fitopts, n_realizations=2),
    }

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"DES": _fitopt_survey_dict(sep, "ALL", 0)}))
    runner = sauron_runner(SimpleNamespace(config=config_path, cheat_cc=False))

    # ALL -> every available fitopt except the fiducial
    fid, sys = runner.parse_fitopt_options(_fitopt_survey_dict(sep, "ALL", 0), "DES")
    assert fid == [0] and sys == [1, 2, 3]
    assert runner.fit_args_dict["fiducial_fitopts"]["DES"] == [0]
    assert runner.fit_args_dict["systematic_fitopts"]["DES"] == [1, 2, 3]
    assert set(runner.fit_args_dict["fitopt_paths"]["DES"]) == {0, 1, 2, 3}
    assert "DUMP_IA" not in runner.fit_args_dict["fitopt_paths"]["DES"][0]

    # Case-insensitive ALL / NONE
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, "all", 0), "DES")[1] == [1, 2, 3]
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, "NONE", 0), "DES")[1] == []
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, "none", 0), "DES")[1] == []
    assert set(runner.fit_args_dict["fitopt_paths"]["DES"]) == {0}

    # Explicit list, as YAML gives "1,3" (a string), and the paths it resolves to
    fid, sys = runner.parse_fitopt_options(_fitopt_survey_dict(sep, "1,3", 0), "DES")
    assert sys == [1, 3]
    data_ia_3 = runner.fit_args_dict["fitopt_paths"]["DES"][3]["DATA_IA"]
    assert len(data_ia_3) == 2
    assert all(p.endswith("FITOPT003.FITRES.gz") for p in data_ia_3)
    assert all(os.path.exists(p) for p in data_ia_3)

    # Single number and YAML list
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, 2, 0), "DES")[1] == [2]
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, [3, 1], 0), "DES")[1] == [1, 3]

    # Fiducial as a list; ALL excludes every fiducial
    fid, sys = runner.parse_fitopt_options(_fitopt_survey_dict(sep, "ALL", [0, 2]), "DES")
    assert fid == [0, 2] and sys == [1, 3]
    fid, sys = runner.parse_fitopt_options(_fitopt_survey_dict(sep, "ALL", "0,2"), "DES")
    assert fid == [0, 2] and sys == [1, 3]

    # Combined SIM_ALL / DATA_ALL layout
    fid, sys = runner.parse_fitopt_options(_fitopt_survey_dict(combined, "ALL", 1), "DES")
    assert fid == [1] and sys == [0, 2, 3]
    assert set(runner.fit_args_dict["fitopt_paths"]["DES"][1]) == {"SIM_ALL", "DATA_ALL"}

    # A fitopt missing for one data realization: explicit request fails, ALL skips it
    os.remove(pathlib.Path(sep["DATA_CC"][1]).parent / "FITOPT003.FITRES.gz")
    with pytest.raises(FileNotFoundError, match="FITOPT003 DATA_CC"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, "1,3", 0), "DES")
    assert runner.parse_fitopt_options(_fitopt_survey_dict(sep, "ALL", 0), "DES")[1] == [1, 2]

    # A missing fiducial file also fails
    with pytest.raises(FileNotFoundError, match="FITOPT003"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, "NONE", 3), "DES")
    # A fitopt that doesn't exist anywhere fails
    with pytest.raises(FileNotFoundError, match="FITOPT007"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, 7, 0), "DES")

    # Both keys are mandatory
    with pytest.raises(ValueError, match="SYSTEMATIC_FITOPTS"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, fiducial=0), "DES")
    with pytest.raises(ValueError, match="FIDUCIAL_FITOPT"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, systematic="ALL"), "DES")
    with pytest.raises(ValueError, match="SYSTEMATIC_FITOPTS and FIDUCIAL_FITOPT"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep), "DES")

    # A systematic can't also be the fiducial
    with pytest.raises(ValueError, match="both FIDUCIAL_FITOPT and SYSTEMATIC_FITOPTS"):
        runner.parse_fitopt_options(_fitopt_survey_dict(sep, "0,1", 0), "DES")

    # Need both sim and data entries
    with pytest.raises(ValueError, match="at least one SIM_\\* and one DATA_\\*"):
        runner.parse_fitopt_options(_fitopt_survey_dict({"SIM_ALL": combined["SIM_ALL"]}, "NONE", 0), "DES")


def test_parse_fitopt_options_wildcard_path(tmp_path):
    """Globbed DATA paths (e.g. .../PIP_..-00*/FITOPT000.FITRES.gz) are checked per matched realization."""
    from runner import sauron_runner

    sim = _make_fitopt_files(tmp_path, "SIM_ALL", [0, 1, 2])
    _make_fitopt_files(tmp_path, "DATA_ALL", [0, 1, 2], n_realizations=3)
    os.remove(tmp_path / "DATA_ALL-0002" / "FITOPT002.FITRES.gz")
    paths = {"SIM_ALL": sim, "DATA_ALL": [str(tmp_path / "DATA_ALL-000*" / "FITOPT000.FITRES.gz")]}

    runner = sauron_runner(SimpleNamespace(config=None, cheat_cc=False))
    assert runner.parse_fitopt_options(_fitopt_survey_dict(paths, "ALL", 0), "DES")[1] == [1]


def test_unpack_dataframes_requires_fitopt_keys(tmp_path):
    """Configs without SYSTEMATIC_FITOPTS / FIDUCIAL_FITOPT fail while the yaml is being parsed."""
    import yaml
    from runner import sauron_runner

    config = yaml.safe_load(open(pathlib.Path(__file__).parent / "test_configs/test_config.yml"))
    del config["DES"]["FIT_OPTIONS"]["SYSTEMATIC_FITOPTS"]
    config_path = tmp_path / "config_missing_fitopts.yml"
    config_path.write_text(yaml.safe_dump(config))

    runner = sauron_runner(SimpleNamespace(config=config_path, cheat_cc=False))
    with pytest.raises(ValueError, match="SYSTEMATIC_FITOPTS must be specified in FIT_OPTIONS for DES"):
        runner.unpack_dataframes()


# ----------------------------------------------------------------------------------------------------
# FITOPT systematic covariance (calculate_fitopt_covariance)
# ----------------------------------------------------------------------------------------------------

def _halve_fitres(src, dst):
    """Write a copy of a FITRES file keeping every other row, in the same whitespace-separated format."""
    df = pd.read_csv(src, comment="#", sep=r"\s+")
    df.iloc[::2].to_csv(dst, sep=" ", index=False, compression="gzip")


def _fitopt_test_config(tmp_path, halve=(), systematic="1", fiducial=0):
    """Build a config from test_config.yml whose SIM/DATA files live in tmp_path as FITOPT000 (symlink to the
    real file) and FITOPT001 (a symlink too, or a copy with half the rows for the keys in `halve`)."""
    import yaml
    config = yaml.safe_load(open(pathlib.Path(__file__).parent / "test_configs/test_config.yml"))
    des = config["DES"]
    for key in ["SIM_IA", "SIM_CC", "DATA_IA", "DATA_CC"]:
        real = des[key]["PATH"]
        d = tmp_path / key
        d.mkdir(parents=True)
        (d / "FITOPT000.FITRES.gz").symlink_to(real)
        if key in halve:
            _halve_fitres(real, d / "FITOPT001.FITRES.gz")
        else:
            (d / "FITOPT001.FITRES.gz").symlink_to(real)
        des[key]["PATH"] = str(d / "FITOPT000.FITRES.gz")
    des["FIT_OPTIONS"]["SYSTEMATIC_FITOPTS"] = systematic
    des["FIT_OPTIONS"]["FIDUCIAL_FITOPT"] = fiducial
    config_path = tmp_path / "config_fitopt.yml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _fitopt_runner(config_path):
    args = SimpleNamespace(config=config_path, cheat_cc=False, sys_cov=True, debug=False, plot=False,
                           skip_cuts=False)
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    return runner


def test_fitopt_covariance_zero_for_identical_fitopt(tmp_path):
    """A fitopt whose files are identical to the fiducial gives zero delta counts and zero covariance."""
    runner = _fitopt_runner(_fitopt_test_config(tmp_path))
    n_bins = len(runner.fit_args_dict["z_bins"]["DES"]) - 1

    cov = runner.calculate_fitopt_covariance("DES", PROB_THRESH=0.5)

    delta = runner.fit_args_dict["fitopt_delta_counts"]["DES"][1]
    assert delta.shape == (n_bins,)
    np.testing.assert_allclose(delta, 0, atol=1e-8)
    np.testing.assert_allclose(cov, 0, atol=1e-8)


def test_fitopt_covariance_shift(tmp_path):
    """Halving the fitopt's data gives a real shift; C = outer(delta, delta); main state is left untouched."""
    runner = _fitopt_runner(_fitopt_test_config(tmp_path, halve=("DATA_IA", "DATA_CC")))
    survey = "DES"
    keys_before = set(runner.datasets)
    lengths_before = {k: len(v.df) for k, v in runner.datasets.items()}

    cov = runner.calculate_fitopt_covariance(survey, PROB_THRESH=0.5)
    delta = runner.fit_args_dict["fitopt_delta_counts"][survey][1]

    # Halving the data roughly halves the decontaminated counts, so delta is clearly negative
    fid_counts = runner._fitopt_counts(survey, 0, 0.5)["n_data"][0]
    assert np.sum(delta) < -0.3 * np.sum(fid_counts)
    np.testing.assert_allclose(cov, np.outer(delta, delta))
    np.testing.assert_allclose(cov, cov.T)
    assert np.min(np.linalg.eigvalsh(cov)) > -1e-6 * np.max(np.abs(cov))

    # The main analysis datasets and fit args are untouched (no cuts applied, no eff_ij/N_gen stored)
    assert set(runner.datasets) == keys_before
    assert {k: len(v.df) for k, v in runner.datasets.items()} == lengths_before
    assert survey not in runner.fit_args_dict["eff_ij"]
    assert survey not in runner.fit_args_dict["N_gen"]


def test_fitopt_covariance_efficiency_frame(tmp_path):
    """If the fitopt's sims lose events along with the data, eff_ij drops too, and the efficiency-frame
    correction removes most of the shift compared with halving only the data."""
    data_only = _fitopt_runner(_fitopt_test_config(tmp_path / "a", halve=("DATA_IA", "DATA_CC")))
    data_only.calculate_fitopt_covariance("DES", PROB_THRESH=0.5)
    delta_data_only = data_only.fit_args_dict["fitopt_delta_counts"]["DES"][1]

    both = _fitopt_runner(_fitopt_test_config(tmp_path / "b", halve=("DATA_IA", "DATA_CC", "SIM_IA", "SIM_CC")))
    both.calculate_fitopt_covariance("DES", PROB_THRESH=0.5)
    delta_both = both.fit_args_dict["fitopt_delta_counts"]["DES"][1]

    assert np.linalg.norm(delta_both) < 0.3 * np.linalg.norm(delta_data_only)


def test_fitopt_covariance_in_calculate_covariance(tmp_path, monkeypatch):
    """calculate_covariance adds the fitopt covariance to cov_sys (other terms stubbed out to zero)."""
    import runner as runner_module
    monkeypatch.setattr(runner_module, "calculate_covariance_matrix_term",
                        lambda func, params, z_bins, *args: np.zeros((len(z_bins) - 1, len(z_bins) - 1)))
    runner = _fitopt_runner(_fitopt_test_config(tmp_path, halve=("DATA_IA",)))
    runner.calculate_covariance(PROB_THRESH=0.5)
    delta = runner.fit_args_dict["fitopt_delta_counts"]["DES"][1]
    np.testing.assert_allclose(runner.fit_args_dict["cov_sys"]["DES"], np.outer(delta, delta))


def test_fitopt_covariance_none_and_multi_fiducial(tmp_path):
    """NONE gives no fitopt covariance; more than one fiducial with systematics raises."""
    runner = _fitopt_runner(_fitopt_test_config(tmp_path / "none", systematic="NONE"))
    assert runner.calculate_fitopt_covariance("DES", PROB_THRESH=0.5) is None

    runner = _fitopt_runner(_fitopt_test_config(tmp_path / "multi", systematic="ALL", fiducial=[0, 1]))
    runner.fit_args_dict["systematic_fitopts"]["DES"] = [1]  # ALL minus both fiducials is empty; force one
    with pytest.raises(ValueError, match="exactly one FIDUCIAL_FITOPT"):
        runner.calculate_fitopt_covariance("DES", PROB_THRESH=0.5)


def test_binned_rate_sys_err(tmp_path, monkeypatch):
    """fit_rate records the per-bin systematic error on the binned rate: sqrt(diag(cov_sys)) / D."""
    import runner as runner_module
    from scipy.sparse import block_diag
    monkeypatch.setattr(runner_module, "calculate_covariance_matrix_term",
                        lambda func, params, z_bins, *args: np.zeros((len(z_bins) - 1, len(z_bins) - 1)))
    runner = _fitopt_runner(_fitopt_test_config(tmp_path, halve=("DATA_IA", "DATA_CC")))
    runner.args.marginalize = False
    runner.parse_global_fit_options()
    survey = "DES"

    runner.calculate_covariance(PROB_THRESH=0.5)
    # Same per-survey steps as sauron.py main
    runner.apply_cuts(survey, subset_version=True)
    runner.get_counts(survey)
    runner.apply_cuts(survey)
    runner.calculate_transfer_matrix(survey)
    runner.load_and_decontaminate_datasets(survey, PROB_THRESH=0.5)
    runner.fit_rate(survey, 1, PROB_THRESH=0.5)

    cov_sys = runner.fit_args_dict["cov_sys"][survey]
    null_counts = runner.fit_args_dict["null_counts"][survey]
    eff_ij = runner.fit_args_dict["eff_ij"][survey]
    D = np.sum(null_counts * eff_ij * runner.fit_args_dict["f_norm"][survey], axis=0)
    sys_err = runner.final_counts[survey]["binned_rate_sys_err"]
    np.testing.assert_allclose(sys_err, np.sqrt(np.diag(cov_sys)) / D)
    assert np.any(sys_err > 0)


# ----------------------------------------------------------------------------------------------------
# Cut systematics (CUT_SYSTEMATICS / calculate_cut_systematics)
# ----------------------------------------------------------------------------------------------------

def test_cut_systematics_covariance(tmp_path):
    """CUT_SYSTEMATICS variants add sum_v outer(Delta_v, Delta_v) to cov_sys, where Delta_v is the variant's
    decontaminated counts mapped onto the nominal efficiency minus the nominal counts. A variant identical to
    the nominal cuts contributes nothing, and the nominal analysis state is left untouched."""
    import copy
    import yaml

    config = yaml.safe_load(open(pathlib.Path(__file__).parent / "test_configs/test_config.yml"))
    nominal_cuts = dict(config["DES"]["CUTS"])
    assert nominal_cuts["x1"].replace(" ", "") == "-3,3", "Test assumes nominal x1 cut of -3, 3."
    tight_c = "-0.2, 0.2"
    # Variant 1 is identical to the nominal x1 cut (Delta must be 0); variant 2 tightens the c cut.
    config["DES"]["CUT_SYSTEMATICS"] = [{"x1": "-3, 3"}, {"c": tight_c}]
    config_path = tmp_path / "config_cut_sys.yml"
    config_path.write_text(yaml.safe_dump(config))

    runner = _fitopt_runner(config_path)
    survey = "DES"
    PROB_THRESH = 0.5

    # Same per-survey steps as sauron.py main
    runner.apply_cuts(survey, subset_version=True)
    runner.get_counts(survey)
    uncut = {k: copy.copy(v) for k, v in runner.datasets.items() if k.startswith(f"{survey}_")}
    runner.apply_cuts(survey)
    runner.calculate_transfer_matrix(survey)

    lengths_before = {k: len(v.df) for k, v in runner.datasets.items()}
    eff_nom = runner.fit_args_dict["eff_ij"][survey].copy()

    cov = runner.calculate_cut_systematics(survey, uncut, PROB_THRESH)

    # Independently compute Delta for the tightened-c variant using the same building blocks.
    N_gen = np.asarray(runner.fit_args_dict["N_gen"][survey], dtype=float)
    n_nom = np.asarray(runner.calculate_CC_contamination(PROB_THRESH, 1, survey), dtype=float)
    tmp = {k: copy.copy(v) for k, v in uncut.items()}
    runner.apply_cuts(survey, datasets=tmp, cuts={**nominal_cuts, "c": tight_c})
    n_v = np.asarray(runner.calculate_CC_contamination(PROB_THRESH, 1, survey, datasets=tmp), dtype=float)
    eff_v = runner.calculate_transfer_matrix(survey, datasets=tmp, store=False)
    P_nom = np.sum(N_gen * eff_nom, axis=0)
    P_v = np.sum(N_gen * eff_v, axis=0)
    assert np.all(P_v > 0)
    delta = n_v * P_nom / P_v - n_nom

    assert np.any(np.abs(delta) > 1e-6), "Tightening the c cut should change the counts."
    # Only the tightened-c variant contributes; the identical-x1 variant adds zero.
    np.testing.assert_allclose(cov, np.outer(delta, delta), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(cov, cov.T)
    assert np.min(np.linalg.eigvalsh(cov)) > -1e-6 * np.max(np.abs(cov))

    # cov_sys started empty (calculate_covariance not called), so it is exactly the cut term.
    np.testing.assert_allclose(runner.fit_args_dict["cov_sys"][survey], cov)
    np.testing.assert_allclose(runner.fit_args_dict["cov_cut_sys"][survey], cov)

    # The nominal datasets and transfer matrix are untouched.
    assert {k: len(v.df) for k, v in runner.datasets.items()} == lengths_before
    np.testing.assert_array_equal(runner.fit_args_dict["eff_ij"][survey], eff_nom)

    # Turned off when systematics are disabled.
    runner.args.sys_cov = False
    assert runner.calculate_cut_systematics(survey, uncut, PROB_THRESH) is None
