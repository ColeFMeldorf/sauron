
# Sauron
from sauron import (unpack_dataframes,
                    calculate_CC_contamination,
                    calculate_covariance_matrix_term,
                    calculate_transfer_matrix,
                    chi2)

# Standard Library
import os
import pathlib
import yaml

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
    os.system(f"python {sauron_path} {config_path} --cheat_cc -o {outpath}")
    results = pd.read_csv(outpath)
    regression_vals = [1.0, 0.0, 0.0]
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-7)  # atol not rtol b/c we expect 0


def test_calc_cov_term():
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    files_input = yaml.safe_load(open(config_path, 'r'))
    datasets, surveys, n_datasets = unpack_dataframes(files_input, corecollapse_are_separate=True)
    survey = "DES"
    z_bins = np.arange(0, 1.4, 0.1)
    cov_mat = calculate_covariance_matrix_term(calculate_CC_contamination, [0.05, 0.1, 0.15], z_bins, datasets, 1,
                                     survey, z_bins, False)
    regression_cov = np.load(pathlib.Path(__file__).parent / "test_cov_term.npy")
    np.testing.assert_allclose(cov_mat, regression_cov, atol=1e-7)


def test_chi():
    config_path = pathlib.Path(__file__).parent / "test_config_5pz.yml"
    files_input = yaml.safe_load(open(config_path, 'r'))
    datasets, surveys, n_datasets = unpack_dataframes(files_input, corecollapse_are_separate=True)
    z_bins = np.arange(0, 1.4, 0.1)
    survey = "DES"
    index = 1
    N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(z_bins)
    eff_ij = calculate_transfer_matrix(datasets[f"{survey}_DUMP_IA"],  datasets[f"{survey}_SIM_IA"], z_bins)
    f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)) / \
                np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
    n_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)
    x = np.array([1.0, 0.0])
    regression_chi = np.load(pathlib.Path(__file__).parent / "test_chi_output.npy")
    np.testing.assert_allclose(chi2(x, N_gen, f_norm, z_bins, eff_ij, n_data), regression_chi, atol=1e-7)