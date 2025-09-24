# Standard Library
import os
import pathlib

import pandas as pd


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def test_regression_specz():
    outpath = pathlib.Path(__file__).parent / "test_regnopz_output.csv"
    os.system(f"python sauron.py tests/test_config.yml -o {outpath}")
    results = pd.read_csv(outpath)
    print(results)
