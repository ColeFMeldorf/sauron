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
    os.system("python sauron.py tests/test_config.yml -o ./test_regnopz_output.csv")
    results = pd.read_csv("./test_regnopz_output.csv")
    print(results)
    
