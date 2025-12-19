
# Sauron
from funcs import chi2, calculate_covariance_matrix_term, power_law, calculate_null_counts
from runner import sauron_runner

# Standard Library
import os
import logging
import pathlib
from types import SimpleNamespace
import subprocess
from scipy.stats import chi2 as scipy_chi2


import numpy as np
import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_regression_specz():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    """
    outpath = pathlib.Path(__file__).parent / "test_regnopz_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regnopz_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression[col], rtol=1e-6)


def test_regression_pz_5datasets():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Photo Zs.
       This also uses 5 datasets rather than 1 to test that functionality.
    """
    outpath = pathlib.Path(__file__).parent / "test_regpz_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regpz_regression.csv")
    for i, col in enumerate(regression.columns):
        if isinstance(regression[col][0], str):
            continue
        logger.debug(f"COL: {col}")
        np.testing.assert_allclose(results[col], regression[col], rtol=1e-6)


def test_perfect_recovery():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (1, 0) with a chi squared of 0.
    """
    outpath = pathlib.Path(__file__).parent / "test_perfect_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_sim.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--cheat_cc", "--no-sys_cov"]
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
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-7)  # atol not rtol b/c we expect 0


def test_perfect_recovery_pz():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (1, 0) with a chi squared of 0. This test includes photo_zs.
    """
    outpath = pathlib.Path(__file__).parent / "test_perfect_output_pz.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "--cheat_cc", "-o", str(outpath), "--no-sys_cov"]
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
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-7)  # atol not rtol b/c we expect 0


def test_calc_cov_term():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"
    runner.z_bins = np.concatenate(([-np.inf], np.arange(0, 1.4, 0.1), [np.inf]))
    cov_mat = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.05, 0.1, 0.15], runner.z_bins, 1,
                                               survey)
    cov_mat = cov_mat[1:-1, 1:-1]  # Exclude first and last bins (infinite bins)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_cov_term.npy")
    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


def test_calc_effij():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config.yml"
    args.config = config_path
    args.cheat_cc = False
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
    eff_ij = eff_ij[1:-1, 1:-1]  # Remove first and last rows/cols corresponding to over/underflow bins
    # Check that it is purely diagonal for this test case
    for i in range(eff_ij.shape[0]):
        for j in range(eff_ij.shape[1]):
            if i != j:
                np.testing.assert_allclose(eff_ij[i, j], 0.0, atol=1e-2)

    config_path = pathlib.Path(__file__).parent / "test_config_pz.yml"
    args.config = config_path
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    survey = "DES"
    eff_ij = runner.calculate_transfer_matrix(survey)[1:-1, 1:-1] # Remove first and last rows/cols corresponding to over/underflow bins
    regression_eff_ij = np.load(pathlib.Path(__file__).parent / "test_effij_regression.npy")
    np.testing.assert_allclose(eff_ij, regression_eff_ij, atol=1e-7)


def test_chi():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    datasets, surveys = runner.unpack_dataframes()
    runner.z_bins = np.concatenate(([-np.inf], np.arange(0, 1.4, 0.1), [np.inf]))

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

    regression_chi = np.load(pathlib.Path(__file__).parent / "test_chi_output.npy")

    np.testing.assert_allclose(chi2(x, null_counts, f_norm, z_centers, eff_ij, n_data, power_law),
                               regression_chi, atol=1e-7)


def test_regression_pz_5datasets_covariance():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Photo Zs.
       This also uses 5 datasets rather than 1 to test that functionality.
    """
    outpath = pathlib.Path(__file__).parent / "test_regpz_sys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regpz_sys_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-3 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression[col], atol=3e-4)
    # The tolerance here is much looser because the inclusion of systematics makes the results more stochastic.
    # The rescale CC for cov now uses deterministic preloaded values via inverse CDF, so the tolerance can be tightened.


def test_coverage_no_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (1, 0) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_coverage_nosys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_coverage.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), '--no-sys_cov']
    # Added --no-sys_cov flag here
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

    a = np.mean(df["alpha_error"]**2)
    b = np.mean(df["beta_error"]**2)
    c = np.mean(df["cov_alpha_beta"])

    mean_cov = np.array([[a, c], [c, b]])

    all_alpha = df["alpha"] - 2.27e-5
    all_beta = df["beta"] - 1.7
    inv_cov = np.linalg.inv(mean_cov)
    all_pos = np.vstack([all_alpha, all_beta])
    product_1 = np.einsum('ij,jl->il', inv_cov, all_pos)
    product_2 = np.einsum("il,il->l", all_pos, product_1)

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.1)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.1)


def test_coverage_with_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (1, 0) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_coverage_sys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_coverage.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]  # Added -c flag here
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

    a = np.mean(df["alpha_error"]**2)
    b = np.mean(df["beta_error"]**2)
    c = np.mean(df["cov_alpha_beta"])

    mean_cov = np.array([[a, c], [c, b]])

    all_alpha = df["alpha"] - 2.27e-5
    all_beta = df["beta"] - 1.7
    inv_cov = np.linalg.inv(mean_cov)
    all_pos = np.vstack([all_alpha, all_beta])
    product_1 = np.einsum('ij,jl->il', inv_cov, all_pos)
    product_2 = np.einsum("il,il->l", all_pos, product_1)

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    logger.info(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.info(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.07)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.07)  # Note the stricter
    # tolerance here. We expect better coverage when systematics are included because they inflate the error bars.


# def test_perfect_recovery_multisurvey():
#     """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
#     Therefore, we should get perfect recovery, i.e. (2.27e-5, 1.7) with a chi squared of 0.
#     This time, we do DES, LOWZ and ROMAN together.
#     """
#     outpath = pathlib.Path(__file__).parent / "test_perfect_output_multisurvey.csv"
#     if os.path.exists(outpath):
#         os.remove(outpath)
#     sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
#     config_path = pathlib.Path(__file__).parent / "test_config_sim_multisurvey.yml"
#     cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--cheat_cc", "--no-sys_cov"]
#     result = subprocess.run(cmd, capture_output=False, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"Command failed with exit code {result.returncode}\n"
#             f"stdout:\n{result.stdout}\n"
#             f"stderr:\n{result.stderr}"
#         )

#     results = pd.read_csv(outpath)
#     regression_vals = [2.27e-5, 1.7, 0.0]
#     for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
#         np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-7)  # atol not rtol b/c we expect 0

# def test_regression_multisurvey():
#     """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
#     This time, we do DES, LOWZ and ROMAN together.
#     """
#     outpath = pathlib.Path(__file__).parent / "test_regmultisurvey_output.csv"
#     if os.path.exists(outpath):
#         os.remove(outpath)
#     sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
#     config_path = pathlib.Path(__file__).parent / "test_config_multisurvey.yml"
#     cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov"]
#     result = subprocess.run(cmd, capture_output=False, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"Command failed with exit code {result.returncode}\n"
#             f"stdout:\n{result.stdout}\n"
#             f"stderr:\n{result.stderr}"
#         )

#     results = pd.read_csv(outpath)
#     regression = pd.read_csv(pathlib.Path(__file__).parent / "test_perfect_output_multisurvey.csv")
#     # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
#     for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
#         np.testing.assert_allclose(results[col], regression[col], rtol=1e-6)