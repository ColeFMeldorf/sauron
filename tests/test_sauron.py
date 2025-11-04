
# Sauron
from sauron import (calculate_covariance_matrix_term,
                    chi2,
                    sauron_runner)

# Standard Library
import os
import pathlib
from types import SimpleNamespace
import subprocess
from scipy.stats import chi2 as scipy_chi2


import numpy as np
import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def test_regression_specz():
    """In this test, we simply test that nothing has changed. This is using CC decontam and realistic data. Spec Zs.
    """
    outpath = pathlib.Path(__file__).parent / "test_regnopz_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config.yml"
    os.system(f"python {sauron_path} {config_path} -o {outpath}")
    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regnopz_regression.csv")
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
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
    os.system(f"python {sauron_path} {config_path} -o {outpath}")
    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regpz_regression.csv")
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
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
    os.system(f"python {sauron_path} {config_path} -o {outpath} --cheat_cc")
    results = pd.read_csv(outpath)
    regression_vals = [1.0, 0.0, 0.0]
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
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
    print(os.system(f"python {sauron_path} {config_path} --cheat_cc -o {outpath}"))
    results = pd.read_csv(outpath)
    regression_vals = [1.0, 0.0, 0.0]
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-7)  # atol not rtol b/c we expect 0


def test_calc_cov_term():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.corecollapse_are_separate = True
    datasets, surveys = runner.unpack_dataframes()
    survey = "DES"
    runner.z_bins = np.arange(0, 1.4, 0.1)
    cov_mat = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.05, 0.1, 0.15], runner.z_bins, 1,
                                               survey)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_cov_term.npy")
    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


def power_law(z, x):
    alpha, beta = x
    return alpha * (1 + z)**beta


def test_chi():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.corecollapse_are_separate = True
    datasets, surveys = runner.unpack_dataframes()
    runner.z_bins = np.arange(0, 1.4, 0.1)
    survey = "DES"
    index = 1
    N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(runner.z_bins)
    eff_ij = runner.calculate_transfer_matrix(survey)
    f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)) / \
        np.sum(datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins))
    n_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
    x = np.array([1.0, 0.0])
    z_centers = 0.5 * (runner.z_bins[1:] + runner.z_bins[:-1])
    regression_chi = np.load(pathlib.Path(__file__).parent / "test_chi_output.npy")
    np.testing.assert_allclose(chi2(x, N_gen, f_norm, z_centers, eff_ij, n_data, power_law),
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
    os.system(f"python {sauron_path} {config_path} -o {outpath} -c")  # Added -c flag here
    results = pd.read_csv(outpath)
    regression = pd.read_csv(pathlib.Path(__file__).parent / "test_regpz_sys_regression.csv")
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
        print("COL: ", col)
        print(results[col])
        print(regression[col])
        np.testing.assert_allclose(results[col], regression[col], atol=8e-3)
    # The tolerance here is much looser because the inclusion of systematics makes the results more stochastic.
    # The rescale CC for cov uses random numbers.


def test_coverage_no_sys():
    """In this test we check the coverage properties of SAURON when there are no systematics.
        We should recover the truth (1, 0) within 1 sigma 68% of the time and within 2 sigma 95% of the time.
    """
    outpath = pathlib.Path(__file__).parent / "test_coverage_nosys_output.csv"
    if os.path.exists(outpath):
        os.remove(outpath)
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_coverage.yml"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath)]
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

    all_alpha = df["delta_alpha"] - 1
    all_beta = df["delta_beta"]
    inv_cov = np.linalg.inv(mean_cov)
    all_pos = np.vstack([all_alpha, all_beta])
    product_1 = np.einsum('ij,jl->il', inv_cov, all_pos)
    product_2 = np.einsum("il,il->l", all_pos, product_1)

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    print("Below 1 sigma:", np.size(sub_one_sigma[0])/np.size(product_2))
    print("Below 2 sigma:", np.size(sub_two_sigma[0])/np.size(product_2))

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
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), '-c']  # Added -c flag here
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

    all_alpha = df["delta_alpha"] - 1
    all_beta = df["delta_beta"]
    inv_cov = np.linalg.inv(mean_cov)
    all_pos = np.vstack([all_alpha, all_beta])
    product_1 = np.einsum('ij,jl->il', inv_cov, all_pos)
    product_2 = np.einsum("il,il->l", all_pos, product_1)

    sub_one_sigma = np.where(product_2 < sigma_1)
    sub_two_sigma = np.where(product_2 < sigma_2)

    print("Below 1 sigma:", np.size(sub_one_sigma[0])/np.size(product_2))
    print("Below 2 sigma:", np.size(sub_two_sigma[0])/np.size(product_2))

    np.testing.assert_allclose(np.size(sub_one_sigma[0])/np.size(product_2), 0.68, atol=0.05)
    np.testing.assert_allclose(np.size(sub_two_sigma[0])/np.size(product_2), 0.95, atol=0.05)  # Note the stricter
    # tolerance here. We expect better coverage when systematics are included because they inflate the error bars.
