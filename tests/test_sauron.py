# Standard Library
import os
import pathlib

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
    print(results)
    regression_vals = [1.10248502, -0.20416661, 0.7775755479255878]
    for i, col in enumerate(["delta_alpha", "delta_beta", "reduced_chi_squared"]):
        print(results[col].values, regression_vals[i])
        np.testing.assert_allclose(results[col], regression_vals[i], rtol=1e-6)


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
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-10)  # atol not rtol b/c we expect 0


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
        np.testing.assert_allclose(results[col], regression_vals[i], atol=1e-10)  # atol not rtol b/c we expect 0
