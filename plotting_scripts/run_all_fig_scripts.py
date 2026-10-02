

# Sauron
from funcs import chi2, calculate_covariance_matrix_term, power_law, calculate_null_counts, \
    rescale_CC_for_cov
from runner import sauron_runner

# Standard Library
import os
import logging
from matplotlib import pyplot as plt
import pathlib
import pytest
from types import SimpleNamespace
import subprocess
from scipy.stats import chi2 as scipy_chi2, ks_2samp, norm


import numpy as np
import pandas as pd
import time
import inspect
import sys

# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SDSS_ONLY_RATE_CONFIG = "/home/colefmeldorf/sauron/config_files/config_SDSS_only_Oct2026.yml"
DES_ONLY_RATE_CONFIG = "/home/colefmeldorf/sauron/config_files/config_DES_only_Oct2026.yml"
SDSS_PLUS_DES_RATE_CONFIG = None


def get_function_name():
    return sys._getframe(2).f_code.co_name


def fetch_from_path_and_check_new(path, starttime, caller):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # check the file is new
    if os.path.getmtime(path) < starttime:
        raise RuntimeError(f"File is not new: {path}")
    else:
        print(f"Fetching {path}")
    # move to ./generated_figures
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"
    os.makedirs(generated_figures_dir, exist_ok=True)
    new_path = generated_figures_dir / os.path.basename(path)
    # Check that the new path isn't already existing
    if "summary_plot.png" in str(new_path):
        new_path = str(new_path).split(".png")[0] + "_" + caller + ".png"
        print(f"That path exists! Using {new_path} instead")

    os.rename(path, new_path)
    path = new_path
    return path


def run_a_cmd(config_path, files_to_get):
    caller = get_function_name()
    print(f"Running command for {caller}")
    outpath_loc = caller + ".csv"
    starttime = time.time()
    outpath = OUTPUT_DIR / outpath_loc
    sauron_path = pathlib.Path(__file__).parent / "../sauron.py"
    cmd = ["python", str(sauron_path), str(config_path), "-o", str(outpath), "--prob_thresh", "0.5", "--plot"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print(f"Successfully ran {get_function_name()}")
    for f in files_to_get:
        fetch_from_path_and_check_new(outpath / f, starttime, caller)


def fig_1_left_3_left_and_8():
    run_a_cmd(DES_ONLY_RATE_CONFIG,
    ["/home/colefmeldorf/sauron/plots/efficiency_matrix_DES.png",
    "/home/colefmeldorf/sauron/summary_plot.png",
    "/home/colefmeldorf/sauron/debug_plots/sanity_check_counts_DES.png"])


def fig_1_right_3_right_and_9():
    run_a_cmd(SDSS_ONLY_RATE_CONFIG,
     ["/home/colefmeldorf/sauron/plots/efficiency_matrix_SDSS.png",
     "/home/colefmeldorf/sauron/summary_plot.png",
     "/home/colefmeldorf/sauron/debug_plots/sanity_check_counts_SDSS.png"])


def fig_4_and_6():
    pass


def run_all_fig_scripts():
    fig_1_left_3_left_and_8()
    fig_1_right_3_right_and_9()
    # Figure 2 is a diagram
    fig_4_and_6()



if __name__ == "__main__":
    run_all_fig_scripts()