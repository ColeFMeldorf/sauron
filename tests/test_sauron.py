# Standard Library
import os
import pathlib

import numpy as np
import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def test_regression_specz():
    outpath = pathlib.Path(__file__).parent / "test_regnopz_output.csv"
    #outpath.unlink()
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config.yml"
    os.system(f"python {sauron_path} {config_path} -o {outpath}")
    results = pd.read_csv(outpath)
    regression_vals = [1.006584, 0.0, 0.643259]
    for i, col in enumerate(["delta_alpha","delta_beta","reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], rtol = 1e-6)
        
def test_perfect_recovery():
    outpath = pathlib.Path(__file__).parent / "test_perfect_output.csv"
    #outpath.unlink()
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    config_path = pathlib.Path(__file__).parent / "test_config_sim.yml"
    os.system(f"python {sauron_path} {config_path} -o {outpath} --cheat_cc")
    results = pd.read_csv(outpath)
    regression_vals = [1.0, 0.0, 0.0]
    for i, col in enumerate(["delta_alpha","delta_beta","reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], atol = 1e-10) # atol not rtol b/c we expect 0
    
