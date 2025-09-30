
# Sauron
from sauron import (calculate_covariance_matrix_term,
                    chi2,
                    sauron_runner)

# Standard Library
import os
import pathlib
from types import SimpleNamespace


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
    datasets, surveys, n_datasets = runner.unpack_dataframes(corecollapse_are_separate=True)
    survey = "DES"
    runner.z_bins = np.arange(0, 1.4, 0.1)
    cov_mat = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.05, 0.1, 0.15], runner.z_bins, 1,
                                               survey)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_cov_term.npy")
    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


def test_chi():
    args = SimpleNamespace()
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    args.config = config_path
    runner = sauron_runner(args)
    datasets, surveys, n_datasets = runner.unpack_dataframes(corecollapse_are_separate=True)
    runner.z_bins = np.arange(0, 1.4, 0.1)
    survey = "DES"
    index = 1
    N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(runner.z_bins)
    eff_ij = runner.calculate_transfer_matrix(survey)
    f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)) / \
                np.sum(datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins))
    n_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
    x = np.array([1.0, 0.0])
    regression_chi = np.load(pathlib.Path(__file__).parent / "test_chi_output.npy")
    np.testing.assert_allclose(chi2(x, N_gen, f_norm, runner.z_bins, eff_ij, n_data), regression_chi, atol=1e-7)