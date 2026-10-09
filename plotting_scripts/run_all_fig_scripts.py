import os
import pathlib
import subprocess
import sys
import numpy as np
import yaml
import pandas as pd
from matplotlib import pyplot as plt
import time
from tests.test_sauron import sauron_coverage_scatterplot
from runner import LaurenNicePlots
from scipy.optimize import nnls
from runner import sauron_runner
from types import SimpleNamespace


OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SDSS_ONLY_RATE_CONFIG = "/home/colefmeldorf/sauron/config_files/config_SDSS_only_Oct2026.yml"
DES_ONLY_RATE_CONFIG = "/home/colefmeldorf/sauron/config_files/config_DES_only_Oct2026.yml"
SDSS_PLUS_DES_RATE_CONFIG = "/home/colefmeldorf/sauron/config_files/config_SDSS_redo_again.yml"
SDSS_PLUS_DES_COVERAGE_CONFIG = "/home/colefmeldorf/sauron/tests/test_configs/test_config_SDSS_DES_coverage.yml"
DES_DATALIKE_SIM = "/project2/rkessler/SURVEYS/" \
    "ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/5_MERGE/MERGE_Dsys_DATADES_SIM_IA/output/" \
    "PIP_D5YR_RATEPZ_NEWCC_SIM_NOMINAL_DATADESSIM_IA-0001/FITOPT000.FITRES.gz"
SDSS_DATALIKE_SIM = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/5_MERGE/"\
    "MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-JH8_SDSS-0001/FITOPT000.FITRES.gz"
POWER_LAW_DTD_CONFIG="/home/colefmeldorf/sauron/config_files/config_SDSS_redo_again_dtd.yml"
APLUSB_CONFIG = "/home/colefmeldorf/sauron/config_files/config_SDSS_redo_again_AplusB.yml"
HOURGLASS_CONFIG = "/home/colefmeldorf/sauron/config_files/config_hourglass_photoz.yml"
HOURGLASS_PROMPT_CONFIG = "/home/colefmeldorf/sauron/config_files/config_hourglass_prompt.yml"
HOURGLASS_PROMPT_BINNED_CONFIG = "/home/colefmeldorf/sauron/config_files/config_hourglass_prompt_binned.yml"


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
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print(f"Successfully ran {get_function_name()}")
    for f in files_to_get:
        fetch_from_path_and_check_new(outpath / f, starttime, caller)

    return outpath


def load_dataframes_and_cut(config_file, survey, base_path_sim):
    files_input = yaml.safe_load(open(config_file, "r"))
    surveys = list(files_input.keys())
    if "FIT_OPTIONS" in surveys:
        surveys.remove("FIT_OPTIONS")

    base_path = files_input[survey]["DATA_ALL"]["PATH"]

    if isinstance(base_path_sim, list):
        base_path_sim = base_path_sim[0]
    if isinstance(base_path, list):
        base_path = base_path[0]

    sim_df = pd.read_csv(base_path_sim, sep = r"\s+", comment="#")
    data_df = pd.read_csv(base_path, sep = r"\s+", comment="#")


    scone_cols = [c for c in data_df.columns if "PROB_SCONE" in c]
    if len(scone_cols) == 1:
        scone_col = scone_cols[0]
    else:
        raise ValueError(f"Could not find a unique PROB_SCONE column in the data, found: {scone_cols}.")

    data_df = data_df[data_df[scone_col] > 0.5]
    data_df = data_df[data_df["x1"] > -3]
    data_df = data_df[data_df["x1"] < 3]
    data_df = data_df[data_df["c"] > -0.3]
    data_df = data_df[data_df["c"] < 0.3]
    data_df = data_df[data_df["FITPROB"] > 0.001]
    sim_df = sim_df[sim_df["x1"] > -3]
    sim_df = sim_df[sim_df["x1"] < 3]
    sim_df = sim_df[sim_df["c"] > -0.3]
    sim_df = sim_df[sim_df["c"] < 0.3]
    sim_df = sim_df[sim_df["FITPROB"] > 0.001]

    return sim_df, data_df


def run_decontamination(config_path, survey):
    LaurenNicePlots()
    args = SimpleNamespace()
    args.config = config_path
    args.cheat_cc = False
    runner = sauron_runner(args)
    runner.z_bins = np.linspace(0.1, 1.0, 11)
    datasets, surveys = runner.unpack_dataframes()

    PROB_THRESH = 0.5

    pulls = []

    pulls = np.empty((50, len(runner.z_bins)-1))
    all_ntrue = np.empty((50, len(runner.z_bins)-1))
    all_ncalc = np.empty((50, len(runner.z_bins)-1))
    for i in range(50):
        index = i+1
        runner.fit_args_dict["z_bins"][survey] = runner.z_bins
        n_calc = runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=False)

        n_true = runner.datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)
        residual = n_true - n_calc
        pull = residual / np.sqrt(n_true)

        pulls[i, :] = pull
        all_ntrue[i, :] = n_true
        all_ncalc[i, :] = n_calc

    pulls = np.array(pulls)

    mean_res = np.mean(all_ntrue - all_ncalc, axis=0)
    std_ntrue = np.std(all_ntrue, axis=0)
    z_centers = (runner.z_bins[:-1] + runner.z_bins[1:]) / 2
    plt.clf()


    plt.errorbar(z_centers, mean_res, yerr=std_ntrue/np.sqrt(50), fmt='o', label='True - Calculated CC Counts')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel('Redshift')
    plt.ylabel('CC Counts')
    plt.legend()


def phot_redshift_plot(config_file, survey, base_path_sim):
    LaurenNicePlots()

    sim_df, data_df = load_dataframes_and_cut(config_file, survey, base_path_sim)
    bins = np.linspace(0, 0.9, 38)
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 3, 1)
    a, _, _ , _= plt.hist2d(sim_df["SIM_ZCMB"], sim_df["zPHOT"], bins=(bins, bins), cmap = "Blues", label = "Sim")
    plt.plot([0, 0.9], [0, 0.9], "k--")
    plt.xlabel("Simulated Redshift")
    plt.ylabel("Photometric Redshift")
    plt.title("Simulated Truth vs Recovered")
    plt.colorbar(label = "Counts")
    plt.subplot(1, 3, 2)
    plt.xlabel("Spectroscopic Redshift")
    b, _, _, _ = plt.hist2d(data_df["HOST_ZSPEC"], data_df["zPHOT"], bins=(bins, bins), cmap = "Reds", label = "Data")
    plt.plot([0, 0.9], [0, 0.9], "k--")
    plt.title("Data Spectroscopic vs Recovered")
    plt.colorbar(label = "Counts")
    plt.subplot(1, 3, 3)

    norm_factor = np.sum(a) / np.sum(b)
    normalized_a = a / norm_factor
    chi = (normalized_a - b) ** 2 / (normalized_a + b)
    chi[np.where((normalized_a + b) == 0)] = 0
    plt.imshow((b - normalized_a).T, origin = "lower", extent = [0, 0.9, 0, 0.9], aspect = "auto", cmap = "seismic", vmin = -30, vmax = 30)
    # vmin = -np.max(np.abs(normalized_a-b)), vmax = np.max(np.abs(normalized_a-b)))
    plt.xlabel("Truth or Spectroscopic Redshift")
    # plt.ylabel("Recovered/phot redshift")
    plt.colorbar(label = "Data - Simulated Counts")
    plt.plot([0, 0.9], [0, 0.9], "k--")


    chi = (normalized_a - b) ** 2 / (normalized_a + b)
    chi = chi[np.where((normalized_a + b) != 0)]

    chi_2 = np.nansum((normalized_a - b) ** 2 / (normalized_a + b))

    plt.title("Data - Simulation")

    # Put the chi squared in a colored box in the bottom right corner
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.text(0.95, 0.05, r"$\chi^2$ = {:.2f}\nReduced $\chi^2$ = {:.2f}".format(chi_2, chi_2 / (len(chi))),
    transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.subplots_adjust(wspace = 0.4)
    plt.tight_layout()
    plt.suptitle("DES Photometric Redshift Performance Compared to Simulation", y = 1.02)
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"
    plt.savefig(generated_figures_dir / f"photometric_redshift_performance_{survey}.png")


def fetch_binned_rate_data(starttime):
    root_path = "/home/colefmeldorf/sauron/plots/"
    DES_root = "binned_rate"
    data_dict = {}
    for ext in ["_DES.npy", "_16_DES.npy", "_84_DES.npy", "_SDSS.npy", "_16_SDSS.npy", "_84_SDSS.npy"]:
        filepath = root_path + DES_root + ext
        # check that the file was made after starttime
        if os.path.exists(filepath):
            if os.path.getmtime(filepath) > starttime:
                data_dict[ext] = np.load(filepath)
            else:
                raise ValueError(f"File {filepath} is older than starttime")
        else:
            raise ValueError(f"Could not find {filepath}")

        print("EXT:", ext)
        print("values:", data_dict.get(ext, None))

    des_median = data_dict.get("_DES.npy", None)
    des_err = (data_dict.get("_84_DES.npy", None) - data_dict.get("_16_DES.npy", None)) / 2

    sdss_median = data_dict.get("_SDSS.npy", None)
    sdss_err = (data_dict.get("_84_SDSS.npy", None) - data_dict.get("_16_SDSS.npy", None)) / 2

    overall_err = np.sqrt(des_err ** 2 + sdss_err ** 2)

    return des_median, des_err, sdss_median, sdss_err, overall_err


def binned_rate_plot(starttime):
    des_median, des_err, sdss_median, sdss_err, overall_err = fetch_binned_rate_data(starttime)
    LaurenNicePlots()
    plt.figure(figsize = (7.5, 5), dpi = 200)
    z_bins = np.linspace(0.05, 0.35, 9)
    z_centers = (z_bins[:-1] + z_bins[1:]) / 2
    z_centers = z_centers[:-1]

    chi_2 = (sdss_median - des_median) ** 2 / overall_err ** 2
    print("Chi-2: ", np.sum(chi_2))
    print("dof: ", len(chi_2))
    print("reduced Chi-2: ", np.sum(chi_2) / len(chi_2))

    plt.errorbar(z_centers, sdss_median, yerr=sdss_err, fmt="o", label="SDSS")
    plt.errorbar(z_centers, des_median, yerr=des_err, fmt="o", label="DES")
    plt.grid()
    plt.legend(loc = "lower right", fontsize = 14)
    z_fine = np.linspace(0.05, 0.4, 100)
    plt.xlabel("Redshift")
    plt.ylabel("Volumetric Rate (SNe yr$^{-1}$ Mpc$^{-3}$)")

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.text(.69, 0.15, r"$\chi^2$ = {:.2f}\nReduced $\chi^2$ = {:.2f}".format(np.sum(chi_2), np.sum(chi_2) / len(chi_2)),
    transform=plt.gca().transAxes, fontsize=14, verticalalignment="bottom", horizontalalignment="right", bbox=props)

    plt.title("Binned Volumetric Rate Comparison between SDSS and DES", fontsize = 16, y=1.08)
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"
    plt.savefig(generated_figures_dir / "binned_rate_comparison.png")


def turnover_power_law(z, x, z_turn=1):
    alpha1, beta1, alpha2, beta2 = x
    fJ = np.where(z < z_turn,
                alpha1 * (1 + z)**beta1,
                alpha2 * (1 + z)**beta2)
    return fJ


def binned_rate_hourglass(df):
    LaurenNicePlots()

    plt.figure(figsize=(10, 5), dpi = 200)

    plt.subplot(1, 2, 1)
    z_bins = np.linspace(0.2, 2.8, 16)
    z_centers = 0.5 * (z_bins[1:] + z_bins[:-1])

    all_results = np.zeros((len(df), len(z_centers)))
    for row in df.iterrows():
        rate_vals = [row[1][f"param_{i}"] for i in range(len(z_centers))]
        all_results[row[0], :] = rate_vals

    mean_rate = np.mean(all_results, axis=0)
    percentiles = np.percentile(all_results, [16, 84], axis=0)

    z_fine = np.linspace(z_centers[0], z_centers[-1], 100)
    plt.plot(z_fine, turnover_power_law(z_fine, [2.27e-5, 1.7, 7.5e-5, -0.1]), label = "Simulated Rate", color = "gray", ls = "--")

    unc_high = percentiles[1] - mean_rate
    unc_low = mean_rate - percentiles[0]
    plt.errorbar(z_centers, mean_rate, yerr=[unc_low, unc_high], fmt="o", color="black", label="Roman Forecast")
    plt.xlabel("Redshift")
    plt.ylabel("Volumetric Rate (SNe yr$^{-1}$ Mpc$^{-3}$)")

    # Redshift values and symmetric errors
    redshift = np.array([0.07, 0.19, 0.33, 0.44, 0.61, 0.81, 1.05, 1.73])
    redshift_err = np.array([0.06, 0.06, 0.08, 0.03, 0.14, 0.07, 0.17, 0.52])

    # R_Ia values with asymmetric errors [lower, upper]
    R_Ia = np.array([0.28, 0.30, 0.38, 0.35, 0.47, 0.60, 0.76, 0.61]) *1e-4
    R_Ia_err_lo = np.array([0.03, 0.02, 0.02, 0.04, 0.03, 0.04, 0.06, 0.10])*1e-4
    R_Ia_err_hi = np.array([0.04, 0.02, 0.02, 0.05, 0.03, 0.04, 0.06, 0.14])*1e-4

    # plt.errorbar expects asymmetric errors as shape (2, N): [lower, upper]
    R_Ia_err = np.array([R_Ia_err_lo, R_Ia_err_hi])

    plt.errorbar(redshift, R_Ia, xerr=redshift_err, yerr=R_Ia_err, fmt="o", label = "S20 Data Compendium")
    plt.legend(loc="lower right")
    ########################################################################

    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.errorbar(redshift, (R_Ia_err_hi + R_Ia_err_lo)/2, xerr=redshift_err, fmt="o", label="S20 Data Compendium")
    mean_err = (unc_high + unc_low) / 2
    plt.errorbar(z_centers, mean_err, xerr  = np.diff(z_centers)[0]/2, fmt="o", color="black", label="Roman Forecast")
    plt.ylabel(r"1$\sigma$ Uncertainty in Rate (SNe yr$^{-1}$ Mpc$^{-3}$)")
    plt.xlabel("Redshift")
    plt.legend()
    plt.ylim(0, 1.5e-5)
    plt.suptitle("Roman Forecast vs. S20 Data Compendium ", fontsize = 16)
    plt.tight_layout()
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"
    plt.savefig(generated_figures_dir / "roman_forecast_vs_s20_data_compendium.png")


from dtd_functions import csfr_double_power_law_uncorrected, build_response_matrix, recover_dtd


def SNR(t, eta_Ia, fP):
    K = 7.132
    # t is measured in Gyr
    rate = np.zeros_like(t)
    rate = np.atleast_1d(rate)  # Ensure rate is at least 1D

    if len(rate) == 1:
        if t < 0.04:
            rate = 0
        elif 0.04 <= t < 0.5:
            rate = eta_Ia * fP * K / (1 - fP)
        else:  # t >= 0.5
            rate = eta_Ia * t**-1
        return rate

    rate[t < 0.04] = 0
    rate[(0.04 <= t) & (t < 0.5)] = eta_Ia * fP * K / (1 - fP)
    rate[t >= 0.5] = eta_Ia * t[t >= 0.5]**-1

    return rate


def binned_dtd_plot(df):
    LaurenNicePlots()
    # z_bins = np.linspace(0.2, 2.8, 16)
    z_sn_edges = np.load(r"/home/colefmeldorf/sauron/plots/z_bin_edges_HOURGLASS.npy")
    z_centers = 0.5 * (z_sn_edges[1:] + z_sn_edges[:-1])
    print("df shape:", df.shape)
    plt.figure(figsize=(4, 4), dpi = 300)

    for k, tau_edges_Gyr in enumerate([[0.04, 0.42, 2.4, 14]]):
        tau_edges_Gyr = np.array(tau_edges_Gyr)
        # covariance_matrix = np.load(r"C:\Users\cmeldorf\Downloads\cov_mat_in_rate_HOURGLASS_.npy")
        # measured_rate = np.load(r"C:\Users\cmeldorf\Downloads\binned_rate_HOURGLASS_new.npy")

        all_results = np.zeros((len(df), len(z_centers)))
        for row in df.iterrows():
            rate_vals = [row[1][f"param_{i}"] for i in range(len(z_centers))]
            all_results[row[0], :] = rate_vals

        print(all_results)
        print("all_results shape:", all_results.shape)
        mean_rate = np.mean(all_results, axis=0)
        percentiles = np.percentile(all_results, [16, 84], axis=0)
        measured_84 = percentiles[1]
        measured_16 = percentiles[0]
        measured_rate = mean_rate



        z_csfr_edges = np.linspace(0.001, 4.0, 400)  # fine grid, edges
        z_csfr_center = 0.5 * (z_csfr_edges[:-1] + z_csfr_edges[1:])

        psi_csfr_peryr = csfr_double_power_law_uncorrected(z_csfr_center)
        psi_csfr = psi_csfr_peryr

        # print("z_sn_edges:", z_sn_edges[:5])
        # print("z_csfr_edges:", z_csfr_edges[:5])
        # print("psi_csfr:", psi_csfr[5])
        # print("tau_edges_Gyr:", tau_edges_Gyr[:5])


        A, t_sn_center = build_response_matrix(
            z_sn_edges, z_csfr_edges, psi_csfr, tau_edges_Gyr
        )

        sigma = (measured_84 - measured_16) / 2
        # print("measured_rate:", measured_rate)
        # print("measured_84", measured_84)
        # print("measured_16", measured_16)
        print("measured rate", np.shape(measured_rate))
        print("sigma", np.shape(sigma))
        print("A shape", np.shape(A))
        Phi, cov_analytic = recover_dtd_weighted(measured_rate, sigma, A, nonnegative=True)


        print("Delay bins (Gyr):     ", list(zip(tau_edges_Gyr[:-1], tau_edges_Gyr[1:])))
        print("Recovered Phi with weights (SNe/Msun/Gyr):", Phi)
        print("Analytic covariance matrix:\n", cov_analytic)
        print("errors on recovered Phi:", np.sqrt(np.diag(cov_analytic)))

        print("Recover dtd no weights")

        Phi_no_weights = recover_dtd(measured_rate, A, nonnegative=True)

        tau_centers = 0.5 * (tau_edges_Gyr[:-1] + tau_edges_Gyr[1:])
        tau_widths = np.diff(tau_edges_Gyr)
        print("tau_centers:", tau_centers)
        plt.errorbar(
            tau_centers, Phi, yerr=np.sqrt(np.diag(cov_analytic)), label="Recovered DTD", lw = 2, color = "C" + str(k)
        )

        print("Recovered phi with weights:", Phi)
        print("Phi errors:", np.sqrt(np.diag(cov_analytic)))

        eta_val = 1.38e-4    # strip the astropy unit once here
        fP = 0.59
        t = np.linspace(0, 10, 100000)

        plt.yscale("log")
        plt.xscale("log")
        plt.xlim(0.04, 10)
        plt.title("Recovered DTD vs True DTD")
        plt.xlabel("Delay time (Gyr)")
        plt.ylabel(r"$\Phi(\tau)$ [SNe / M$_\odot$ / Gyr]")
        plt.legend()

        z_sn_edges_fine = np.linspace(0, 3, 100)

        psi_csfr_peryr = csfr_double_power_law_uncorrected(z_csfr_center)
        psi_csfr = psi_csfr_peryr
        A, t_sn_center = build_response_matrix(
            z_sn_edges, z_csfr_edges, psi_csfr, tau_edges_Gyr
        )

        simulated_rate = SNR(t_sn_center, eta_val, fP) * 1e-9  # convert to SNe/yr/Mpc^3
        sigma = 0.001 * simulated_rate  # 10% uncertainty
        Phi, cov_analytic = recover_dtd_weighted(measured_rate, sigma, A, nonnegative=True)
        for t in range(len(tau_edges_Gyr)-1):
            tau_lo = tau_edges_Gyr[t]
            tau_hi = tau_edges_Gyr[t+1]
            print(f"phi t {Phi[t]}")
            if t == 0:
                label = "True DTD binned"
            else:
                label = None
            plt.plot([tau_lo, tau_hi], [Phi[t], Phi[t]], "--", color = "C" + str(k), lw = 1, zorder = 10, label = label)

    plt.legend()
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"
    plt.savefig(generated_figures_dir / "roman_forecast_binned_dtd.png")
    print("saved to", generated_figures_dir / "roman_forecast_binned_dtd.png")


def recover_dtd_weighted(R_sn_peryr, sigma_R_peryr, A, nonnegative=True):
    """ Same as recover_dtd, but downweights noisy SN rate bins using their
    reported uncertainty sigma_R_peryr (same units as R_sn_peryr).

    Internally this "whitens" the system: divide every row of A and the
    corresponding entry of R by sigma_i, which turns weighted least
    squares into ordinary least squares on the rescaled system. This is
    also the correct way to feed uncertainty into nnls, since nnls has
    no native `sigma=` argument.

    Returns
    -------
    Phi        : recovered DTD, SNe/Msun/Gyr
    cov_analytic : (n_dtd, n_dtd) covariance matrix, valid ONLY for the
                   unconstrained (nonnegative=False) solution, or for the
                   nonnegative solution if you've separately confirmed no
                   bin is sitting at the Phi_j = 0 boundary.
    """
    R_perGyr = R_sn_peryr * 1e9
    sigma_perGyr = sigma_R_peryr * 1e9

    # plt.errorbar(np.linspace(0,1,len(R_perGyr)), R_perGyr, yerr=sigma_perGyr, fmt='o')
    # plt.savefig("R_perGyr_vs_sigma_perGyr.png")
    # plt.close()

    # whiten: divide each equation by its uncertainty
    A_w = A / sigma_perGyr[:, None]
    R_w = R_perGyr / sigma_perGyr

    if nonnegative:
        Phi, resid = nnls(A_w, R_w)
    else:
        Phi, *_ = np.linalg.lstsq(A_w, R_w, rcond=None)
    print("Phi:", Phi)
    # analytic covariance: (A_w^T A_w)^-1  (equivalent to (A^T C^-1 A)^-1)
    cov_analytic = np.linalg.inv(A_w.T @ A_w)

    return Phi, cov_analytic


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
    outpath = run_a_cmd(SDSS_PLUS_DES_COVERAGE_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])
    generated_figures_dir = pathlib.Path(__file__).parent / "generated_figures"

    df = pd.read_csv(outpath)
    df = df[df["survey"].str.contains("combined")]
    sauron_coverage_scatterplot(df,
    outpath=generated_figures_dir / "test_coverage_sys_scatter_SDSS_plus_DES.png")



def fig_5():
    phot_redshift_plot(DES_ONLY_RATE_CONFIG, "DES", base_path_sim = DES_DATALIKE_SIM)
    phot_redshift_plot(SDSS_ONLY_RATE_CONFIG, "SDSS", base_path_sim = SDSS_DATALIKE_SIM)


def fig_7():
    run_a_cmd("/home/colefmeldorf/sauron/config_files/config_SDSS_plus_DES_lowbins_Oct2026.yml", ["/home/colefmeldorf/sauron/summary_plot.png"])
    binned_rate_plot(0)


def fig_10():
    run_a_cmd(SDSS_PLUS_DES_RATE_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])


def fig_11():
    run_a_cmd(POWER_LAW_DTD_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])


def fig_12():
    run_a_cmd(APLUSB_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])


def fig_13():
    outpath = run_a_cmd(HOURGLASS_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])
    binned_rate_hourglass(pd.read_csv(outpath))


def fig_14():
    run_a_cmd(HOURGLASS_PROMPT_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"])


def fig_15():
    outpath = run_a_cmd(HOURGLASS_PROMPT_BINNED_CONFIG, ["/home/colefmeldorf/sauron/summary_plot.png"] )
    binned_dtd_plot(pd.read_csv(outpath))


def run_all_fig_scripts():
    # Figure 2 is a diagram

    fig_1_left_3_left_and_8()
    fig_1_right_3_right_and_9()
    fig_4_and_6()
    fig_5()
    fig_7()
    fig_10()
    fig_11()
    fig_12()
    fig_13()
    fig_14()
    fig_15()



if __name__ == "__main__":
    run_all_fig_scripts()
