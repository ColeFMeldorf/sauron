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
   # outpath.unlink()
    os.system(f"python ../sauron.py ./test_config.yml -o {outpath}")
    results = pd.read_csv(outpath)
    regression_vals = [1.006584, 0.0, 0.643259]
    for i, col in enumerate(["delta_alpha","delta_beta","reduced_chi_squared"]):
        np.testing.assert_allclose(results[col], regression_vals[i], rtol = 1e-6)
