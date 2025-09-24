# Standard Library
import os
import yaml


import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binned_statistic as binstat


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

def test_regression_specz():
    os.system("python sauron.py tests/test_config.yml > tests/test_regnopz_output.log")
    with open("tests/test_regnopz_output.log", "r") as f:
        lines = f.readlines()
    string = "Delta Alpha and Delta Beta: [1.00658416 0.        ]"
    string2 = "Reduced Chi Squared: 0.6432586111413626"
    assert string in lines
    assert string2 in lines
