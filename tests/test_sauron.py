
# Sauron
from funcs import chi2, calculate_covariance_matrix_term, power_law, calculate_null_counts, chi2_unsummed
from runner import sauron_runner

# Standard Library
import os
import logging
from matplotlib import pyplot as plt
import pathlib
import pytest
from types import SimpleNamespace
import subprocess
from scipy.stats import chi2 as scipy_chi2


import numpy as np
import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


# Configure the basic logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov", "--prob_thresh", "0.13"]
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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov", "--prob_thresh", "0.5"]
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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--cheat_cc", "--no-sys_cov",
           "--prob_thresh", "0.13"]
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
    cmd = ["python", str(sauron_path), str(config_path), "--cheat_cc", "-o", str(outpath), "--no-sys_cov",
           "--prob_thresh", "0.13"]
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
    runner.z_bins = np.arange(0, 1.4, 0.1)
    cov_mat = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.45, 0.5, 0.55], runner.z_bins, 1,
                                               survey)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_cov_term.npy")
    plot = True
    if plot:
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(cov_mat, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Covariance Term Value')
        plt.title('Calculated Covariance Matrix Term')
        plt.xlabel('Redshift Bin Index')
        plt.ylabel('Redshift Bin Index')
        plt.subplot(1,2,2)
        plt.imshow(regression_cov, origin='lower', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Covariance Term Value')
        plt.title('Regression Covariance Matrix Term')
        plt.xlabel('Redshift Bin Index')
        plt.ylabel('Redshift Bin Index')
        plt.savefig(pathlib.Path(__file__).parent / "test_cov_term.png")

    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


def test_calc_effij():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config.yml"
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

    config_path = pathlib.Path(__file__).parent / "test_config_pz.yml"
    args.config = config_path
    runner = sauron_runner(args)
    runner.unpack_dataframes()
    survey = "DES"
    eff_ij = runner.calculate_transfer_matrix(survey)
    eff_ij = eff_ij[1:-1, :]
    regression_eff_ij = np.load(pathlib.Path(__file__).parent / "test_effij_regression.npy")
    np.testing.assert_allclose(eff_ij, regression_eff_ij, atol=1e-7)


def test_chi():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
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

    regression_chi = np.load(pathlib.Path(__file__).parent / "test_chi_output.npy")

    np.testing.assert_allclose(chi2_unsummed(x, null_counts, f_norm, z_centers, eff_ij, n_data, power_law),
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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), '--no-sys_cov', "--prob_thresh", "0.5"]
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
        plt.savefig(pathlib.Path(__file__).parent / "test_coverage_nosys_hist.png")

    logger.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    logging.debug(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logging.debug(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.1)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.1)


def test_coverage_with_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (2.27e-5, 1.7) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_coverage_sys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_coverage.yml"
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

    plot = True
    if plot:
        import matplotlib.pyplot as plt

        plt.clf()

        plt.hist(product_2, bins=10, density=True, alpha=0.7, color='blue', label='Observed')
        x = np.linspace(0, 12, 100)
        # Dof = 6, 8 bins - 2 fitted parameters
        plt.plot(x, scipy_chi2.pdf(x, 2), color='red', linestyle='dashed', label='Expected')
        plt.axvline(sigma_1, color='r', linestyle='dashed', linewidth=1)
        plt.axvline(sigma_2, color='g', linestyle='dashed', linewidth=1)
        plt.xlabel("Chi-squared statistic")
        plt.savefig(pathlib.Path(__file__).parent / "test_coverage_sys_hist.png")

    logger.info(f"Below 1 sigma: {np.size(sub_one_sigma[0])/np.size(product_2)}")
    logger.info(f"Below 2 sigma: {np.size(sub_two_sigma[0])/np.size(product_2)}")

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.05)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.05)  # Note the stricter
    # tolerance here. We expect better coverage when systematics are included because they inflate the error bars.


def test_perfect_recovery_multisurvey():
    """In this test, we use the simulation as data (eliminating shot noise) and skip CC decontam.
    Therefore, we should get perfect recovery, i.e. (2.27e-5, 1.7) with a chi squared of 0.
    This time, we do DES, LOWZ and ROMAN together.
    """
    outpath = pathlib.Path(__file__).parent / "test_perfect_output_multisurvey.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_sim_multisurvey.yml"
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


def test_regression_multisurvey():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    This time, we do DES, LOWZ and ROMAN together.
    """
    outpath = pathlib.Path(__file__).parent / "test_regmultisurvey_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_multisurvey.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--no-sys_cov"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regmultisurvey_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression[col], rtol=1e-6)


def test_apply_cut():
    """Test that apply_cuts method in sauron.py works as expected."""
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_cuts.yml"
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
    outpath = pathlib.Path(__file__).parent / "test_desdatareg_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "../config_des_data_zphot.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_desdatareg_regression.csv")
    # Updated from delta alpha and delta beta to just alpha beta. Difference ~10^-4 level.
    for i, col in enumerate(["alpha", "beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression[col], rtol=1e-6)



def test_cc_decontam():
    config_path = pathlib.Path(__file__).parent / "test_config_coverage.yml"
    #config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args = SimpleNamespace()
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    #runner.z_bins = np.arange(0.1, 1.0, 0.1)
    runner.z_bins = np.linspace(0.1, 1.0, 8)
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"

    PROB_THRESH = 0.5
    logger.debug("successfully reran")

    pulls = []

    pulls = np.empty((50, len(runner.z_bins)-1))
    all_ntrue = np.empty((50, len(runner.z_bins)-1))
    all_ncalc = np.empty((50, len(runner.z_bins)-1))
    for i in range(50):
        index = i+1
        logger.debug(f"Working on survey {survey}, dataset {index} -------------------")
        runner.fit_args_dict['z_bins'][survey] = runner.z_bins
        n_calc = runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=True)

        n_true = runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
        residual = n_true - n_calc
        pull = residual / np.sqrt(n_true)

        pulls[i,:] = pull
        all_ntrue[i,:] = n_true
        all_ncalc[i,:] = n_calc
        #pulls.extend(list(pull))

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
    plt.errorbar(z_centers, mean_ntrue, yerr=std_ntrue, fmt='o', label='True CC Counts')
    plt.errorbar(z_centers, mean_ncalc, yerr=std_ncalc, fmt='o', label='Calculated CC Counts')
    plt.xlabel('Redshift')
    plt.ylabel('CC Counts')
    plt.savefig(pathlib.Path(__file__).parent / "test_cc_decontam_counts.png")


    np.testing.assert_allclose(means, 0.0, atol=1/np.sqrt(50))

