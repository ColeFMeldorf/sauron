
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pippin_paths import build_pippin_paths


load_dump = False
scone_cut_thresh = 0.0
inverse = False
if scone_cut_thresh == 0 and inverse:
    raise ValueError("scone_cut_thresh cannot be 0 when inverse is True")


def get_scone_col(df):
    potentials = []
    for c in df.columns:
        if "SCONE" in c:
            potentials.append(c)

    if len(potentials) == 1:
        return potentials[0]
    else:
        if "PROB_SCONE" in potentials:
            return "PROB_SCONE"
        raise ValueError(f"something went wrong. found {potentials}")


def scone_cut(df, scone_col = None, scone_cut_thresh = None, inverse = False):
    print(f"Applying scone cut with threshold {scone_cut_thresh}")
    if scone_col is None:
        scone_col = get_scone_col(df)
    if inverse:
        return df[df[scone_col] <= scone_cut_thresh]
    return df[df[scone_col] > scone_cut_thresh]

def apply_lc_cuts(df):
    print("applying LC cuts")
    df = df.iloc[np.where(df["x1"] < 3)]
    df = df.iloc[np.where(df["c"] < 0.3)]
    df = df.iloc[np.where(df["x1"] > -3)]
    df = df.iloc[np.where(df["c"] > -0.3)]
    df = df.iloc[np.where(df["FITPROB"] > 0.001)]


    df.loc[df["RA"] < 0, "RA"] += 360
    return df


def scone_plot(df_sim_all, df_data, survey = "", df_sim_ia = None, df_sim_cc = None):
    bins = np.linspace(scone_cut_thresh, 1, 20)
    plt.hist(scone_cut(df_sim_all, scone_cut_thresh=scone_cut_thresh)[get_scone_col(df_sim_all)], label = "All Sim",
        bins = bins, histtype = "step", density = True)
    plt.hist(scone_cut(df_data, scone_cut_thresh=scone_cut_thresh)[get_scone_col(df_data)], label = "All Real Data",
        bins = bins, histtype = "step", density = True)
    # if df_sim_ia is not None:
    #     plt.hist(df_sim_ia[get_scone_col(df_sim_ia)], label = "Sim Ia Data", bins = bins, histtype = "step")
    # if df_sim_cc is not None:
    #     plt.hist(df_sim_cc[get_scone_col(df_sim_cc)], label = "Sim CC Data", bins = bins, histtype = "step")
    plt.legend()
    plt.yscale("log")
    plt.title(f"SCONE score Sim and Data {survey} with PIa > {scone_cut_thresh}")


def real_sim_chisq(real_counts, sim_counts, sim_scale):
    data_err = np.sqrt(real_counts)
    sim_err = np.sqrt(sim_counts) * sim_scale

    data_err[np.where(data_err == 0)] = 1
    sim_err[np.where(sim_err == 0)] = 1
    err = np.sqrt(sim_err**2 + data_err**2)
    chisq = (real_counts - sim_counts)**2 / err**2
    dof = len(chisq)
    chisq = np.nansum(chisq)
    return chisq, dof

def SNRMAX_plot(df_sim_all, df_data, survey = "", column = 0):
    # plt.suptitle(survey)
    for i, snrmaxindex in zip(np.array([1, 3, 5]) + column, np.array([1, 2, 3])):
        plt.subplot(3, 2, i)
        print(f"subplot number 6, column {column + 1}, row {column * 3 + i+1}")
        plt.title(f"{survey} SNRMAX{snrmaxindex}, PIa > {scone_cut_thresh}")



        sim_data = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)[f"SNRMAX{snrmaxindex}"]
        real_data = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)[f"SNRMAX{snrmaxindex}"]
        weights = np.full_like(sim_data, real_data.shape[0]/sim_data.shape[0], dtype=float)

        bins = np.linspace(0, 100, 20)
        sim_counts, bins, _ = plt.hist(sim_data, weights = weights,
            alpha=0.5, label=f"Sim All rescale: {weights[0]:.2f}", bins = bins)
        real_counts, bins, _ = plt.hist(real_data,
            alpha=0.5, label="Data All", bins = bins)

        chisq, dof = real_sim_chisq(real_counts, sim_counts, weights[0])
        plt.title(f"Chi^2 = {chisq:.3f} dof = {dof:.1f}")

        plt.text(0.95, 0.95, f"Min SIM SNRMax {np.min(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)[f'SNRMAX{snrmaxindex}'])}\nMin DATA SNRMax {np.min(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)[f'SNRMAX{snrmaxindex}'])}", ha='right', va='top', transform=plt.gca().transAxes)
        plt.legend()
        plt.yscale("log")
    # plt.tight_layout()


# def mb_plot(df_sim_all, df_data, survey = "", column = 0):
#     plt.subplot(1, 2, column + 1)
#     sign = ">" if not inverse else "<="

#     plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")

#     plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], density = True, alpha=0.5, label="Sim All")
#     plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], density = True, alpha=0.5, label="Data All")
#     plt.legend()
#     plt.yscale("log")
#     plt.xlabel("mB")
#     plt.ylabel("Density")


def mb_plot_split_on_type(df_sim_cc, df_sim_ia, df_data, survey = "", column = 0):
    plt.subplot(1, 2, column + 1)
    plt.title(f"{survey}PIa > {scone_cut_thresh}")

    plt.hist(df_sim_cc["mB"], density = True, alpha=0.5, label="Sim CC")
    plt.hist(df_sim_ia["mB"], density = True, alpha=0.5, label="Sim IA")
    plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["mB"], density = True, alpha=0.5, label="Data All")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("mB")
    plt.ylabel("Density")


# def mb_err_plot(df_sim_all, df_data, survey = "", column = 0):
#     plt.subplot(1, 2, column + 1)
#     sign = ">" if not inverse else "<="
#     plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")

#     plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)["mBERR"], density = True, alpha=0.5, label="Sim All")
#     plt.hist(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["mBERR"], density = True, alpha=0.5, label="Data All")
#     plt.legend()
#     plt.yscale("log")
#     plt.xlabel("mBERR")
#     plt.ylabel("Density")


import astropy.units as u
from astropy.cosmology import Planck18

# Calculate distance modulus at redshift z = 0.5
mb = -19.4

def distance_modulus(df, alpha = 0.14, beta = 3.0):
    return df["mB"] - mb - alpha * df["x1"] + beta * df["c"]

def hubble_residual(df, alpha = 0.14, beta = 3.0):
    return distance_modulus(df, alpha, beta) - Planck18.distmod(df["zHD"]).value

def hubble_residual_plot(df_sim_all, df_data, survey = "", column = 0):
    plt.subplot(1, 2, column + 1)
    sign = ">" if not inverse else "<="
    plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")

    sim_data = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)
    real_data = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)

    sim_HR = hubble_residual(sim_data)
    real_HR = hubble_residual(real_data)

    weights = np.full_like(sim_HR, real_HR.shape[0]/sim_HR.shape[0], dtype=float)
    bins = 20
    ###
    sim_counts, bins, _ = plt.hist(sim_HR,  label=f"Sim All, rescale: {weights[0]:.2f}", bins=bins, weights = weights)

    real_counts, bins = np.histogram(real_HR, bins=bins)
    plt.errorbar((bins[:-1] + bins[1:]) / 2, real_counts, yerr=np.sqrt(real_counts), fmt='o', label="Data All", color="black")
    chisq, dof = real_sim_chisq(real_counts, sim_counts, weights[0])
    plt.title(f"Chi^2 = {chisq:.3f} dof = {dof:.1f}")
    ###
    #plt.hist(sim_HR, alpha=0.5, label="Sim All", weights=weights)
    #plt.hist(real_HR, alpha=0.5, label="Data All")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("HUBBLE_RESIDUAL")
    plt.ylabel("Density")


def simple_histogram_plot(df_sim_all, df_data, col, survey = "", plot_column = 0):

    # if col == "PROB_SCONE":
    #     # sim_all_scone_col = get_scone_col(df_sim_all)
    #     # df_sim_all["SCONE"] = df_sim_all[sim_all_scone_col]
    #     data_scone_col = get_scone_col(df_data)
    #     df_data["PROB_SCONE"] = df_data[data_scone_col]


    plt.subplot(1, 2, plot_column + 1)
    sign = ">" if not inverse else "<="
    plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")

    if col == "c":
        bins = np.linspace(-0.5, 0.5, 20)
    elif col == "x1":
        print("Setting x1 bins")
        bins = np.linspace(-3, 3, 20)
    elif "scone" in col.lower():
        if inverse:
            bins = np.linspace(0, scone_cut_thresh, 10)
        else:
            if scone_cut_thresh <= 0.1:
                bins = np.linspace(scone_cut_thresh, 1, 20)
            else:
                bins = np.linspace(scone_cut_thresh, 1, 10)

    else:
        bins = 20



    df_sim_ia = df_sim_all[np.isin(df_sim_all["SIM_GENTYPE"], [1])]
    df_sim_cc = df_sim_all[~np.isin(df_sim_all["SIM_GENTYPE"], [1])]


    sim_data = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    real_data  = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]

    sim_data_ia = scone_cut(df_sim_ia, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    sim_data_cc = scone_cut(df_sim_cc, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]

    weights = np.full_like(sim_data, real_data.shape[0]/sim_data.shape[0], dtype=float)

    sim_counts, _, _ = plt.hist(sim_data,  label=f"Sim All, rescale: {weights[0]:.2f}", bins=bins, weights = weights, histtype = "step")

    weights = np.full_like(sim_data_ia, real_data.shape[0]/sim_data.shape[0], dtype=float)
    plt.hist(sim_data_ia, label="Sim Ia", bins=bins, weights = weights, histtype = "step")
    print(sim_data_cc)
    weights = np.full_like(sim_data_cc, real_data.shape[0]/sim_data.shape[0], dtype=float)
    print("WEIGTHS", weights)
    plt.hist(sim_data_cc, label="Sim CC", bins=bins, weights = weights, histtype = "step")
    #real_counts, bins, _ = plt.hist(real_data, alpha=0.5, label="Data All", bins=bins)

    real_counts, bins = np.histogram(real_data, bins=bins)
    plt.errorbar((bins[:-1] + bins[1:]) / 2, real_counts, yerr=np.sqrt(real_counts), fmt='o', label="Data All", color="black")
    chisq, dof = real_sim_chisq(real_counts, sim_counts, weights[0])
    plt.title(f"Chi^2 = {chisq:.3f} dof = {dof:.1f}")

    num_ia = sim_data_ia.shape[0]
    num_cc = sim_data_cc.shape[0]

    plt.text(0.95, 0.95, f"Sim Ia: {num_ia}, Sim CC: {num_cc}", transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

    print(f"Reduced Chi^2 for survey {survey} for thing {col} is {chisq/dof:.3f}")

    plt.legend()
    plt.yscale("log")
    plt.xlabel(col)
    plt.ylabel("Counts")


def fit_cc_scale(n_ia, n_cc, n_data, w):
    """Find the CC scale factor s minimizing chi^2 of data vs. w*n_ia + w*s*n_cc.
    Returns (s_best, chi^2 at s_best, chi^2 at s=1)."""
    from scipy.optimize import minimize_scalar

    def chisq(s):
        model = w * n_ia + w * s * n_cc
        var = n_data + w**2 * n_ia + (w * s)**2 * n_cc
        var = np.where(var == 0, 1, var)
        return np.nansum((n_data - model)**2 / var)

    res = minimize_scalar(chisq, bounds = (0, 50), method = "bounded")
    return res.x, chisq(res.x), chisq(1.0)


def cc_rescaled_histogram_plot(df_sim_all, df_data, col = "PROB_SCONE", survey = "", plot_column = 0):
    """Like simple_histogram_plot, but the CC component gets an extra scale factor s
    (on top of the global N_data/N_sim_all normalization) chosen to minimize chi^2."""
    plt.subplot(1, 2, plot_column + 1)

    df_sim_ia = df_sim_all[np.isin(df_sim_all["SIM_GENTYPE"], [1])]
    df_sim_cc = df_sim_all[~np.isin(df_sim_all["SIM_GENTYPE"], [1])]

    sim_data = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    real_data = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    sim_data_ia = scone_cut(df_sim_ia, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    sim_data_cc = scone_cut(df_sim_cc, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]

    if col == "c":
        bins = np.linspace(-0.5, 0.5, 20)
    elif col == "x1":
        bins = np.linspace(-3, 3, 20)
    else:
        bins = np.histogram_bin_edges(sim_data, bins = 10)

    n_ia, _ = np.histogram(sim_data_ia, bins = bins)
    n_cc, _ = np.histogram(sim_data_cc, bins = bins)
    n_data, _ = np.histogram(real_data, bins = bins)

    w = real_data.shape[0] / sim_data.shape[0]

    s_best, chi_best, chi_1 = fit_cc_scale(n_ia, n_cc, n_data, w)
    dof = len(n_data)

    ia_counts = w * n_ia
    cc_counts = w * s_best * n_cc
    plt.stairs(ia_counts + cc_counts, bins, label = f"Sim Ia + CC (CC x {s_best:.3f})", lw = 2)
    plt.stairs(ia_counts, bins, label = f"Sim Ia (norm {w:.2f})")
    plt.stairs(cc_counts, bins, label = f"Sim CC (norm {w:.2f} x {s_best:.3f})")
    plt.errorbar((bins[:-1] + bins[1:]) / 2, n_data, yerr = np.sqrt(n_data), fmt = 'o', label = "Data All", color = "black")

    sign = ">" if not inverse else "<="
    plt.title(f"{survey} PIa {sign} {scone_cut_thresh}\n"
              f"Chi^2 = {chi_best:.2f} (s=1: {chi_1:.2f}), dof = {dof}")
    plt.text(0.95, 0.95, f"CC rescale = {s_best:.3f}", transform = plt.gca().transAxes,
             verticalalignment = 'top', horizontalalignment = 'right')

    print(f"{survey} {col}: best CC rescale = {s_best:.3f}, chi^2 = {chi_best:.3f} "
          f"(chi^2 at s=1: {chi_1:.3f}), dof = {dof}")

    plt.legend(loc = "upper left")
    plt.yscale("log")
    plt.xlabel(col)
    plt.ylabel("Counts")
    plt.ylim(0, 2e3)
    return s_best


def blank_versus_blank_scatter_plot(df_sim_all, df_data, col1, col2, survey = "", plot_column = 0, col3 = None):

    #df_sim_all = apply_lc_cuts(df_sim_all)
    #df_data = apply_lc_cuts(df_data)

    plt.subplot(1, 2, plot_column + 1)
    sign = ">" if not inverse else "<="
    plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")

    print(f"Data range for {col1}: {np.min(df_data[col1])} to {np.max(df_data[col1])}")
    print(f"Data range for {col2}: {np.min(df_data[col2])} to {np.max(df_data[col2])}")
    print(f"Sim range for {col1}: {np.min(df_sim_all[col1])} to {np.max(df_sim_all[col1])}")
    print(f"Sim range for {col2}: {np.min(df_sim_all[col2])} to {np.max(df_sim_all[col2])}")

    sim_data_1 = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col1]
    real_data_1  = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col1]


    sim_data_2 = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col2]
    real_data_2 = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col2]

    if col3 is not None:
        sim_data_3 = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col3]
        real_data_3 = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col3]

        color1 = sim_data_3 / np.max(sim_data_3)
        color2 = real_data_3 / np.max(real_data_3)

    else:
        color1 = "blue"
        color2 = "red"


    plt.scatter(sim_data_1, sim_data_2, alpha=0.3, label="Sim All", c=color1, s=5, marker = "s")
    plt.scatter(real_data_1, real_data_2, alpha=0.3, label="Data All", c=color2, s=5)
    if col3 is not None:
        plt.colorbar(label=col3)
    if np.max(real_data_1) > 100:
        plt.xscale("log")
    if np.max(real_data_2) > 100:
        plt.yscale("log")
    plt.legend()
    plt.xlabel(col1)
    plt.ylabel(col2)


def data_versus_data_histogram_plot(df1, survey1, df2, survey2, col, label = None):
    sign = ">" if not inverse else "<="
    plt.title(label  + f" PIa {sign} {scone_cut_thresh}")

    if col == "c":
        bins = np.linspace(-0.5, 0.5, 30)
    else:
        bins = 20

    df1_data = scone_cut(df1, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    df2_data  = scone_cut(df2, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]


    plt.hist(df1_data, label=f"{survey1}", bins=bins if col == "c" else None, histtype = "step", lw = 3)
    plt.hist(df2_data, label=f"{survey2}", bins=bins if col == "c" else None, histtype = "step", lw = 3)
        # try:
        #     plt.hist(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["SIM_"+col], density = True, alpha=0.5, label="Sim All, Generated", bins = bins)
        # except KeyError:
        #     pass

    plt.legend()
    plt.yscale("log")
    plt.xlabel(col)
    plt.ylabel("Counts")


def mB_as_a_function_of_z(df_sim_all, df_data, survey = "", column = 0):
    plt.subplot(1, 2, column + 1)
    sign = ">" if not inverse else "<="

    plt.title(f"{survey}PIa {sign} {scone_cut_thresh}")


    from scipy.stats import binned_statistic

    all_z = scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["zHD"]
    print(all_z)
    min_z  = np.nanmin(all_z)
    max_z  = np.nanmax(all_z)
    print("min, max", min_z, max_z)
    z_bins = np.arange(min_z, max_z+0.05, 0.05)

    stat, edges, _ = binned_statistic(all_z, scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], statistic='mean', bins=z_bins)
    std, edges, _ = binned_statistic(all_z, scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], statistic='std', bins=z_bins)

    centers = (edges[1:] + edges[:-1]) / 2
    plt.plot(centers, stat, label="Sim All Mean", color="blue")
    #plt.fill_between(centers, stat - std, stat + std, color="blue", alpha=0.2)


    all_data_z = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["zHD"]

    stat_data, edges_data, _ = binned_statistic(all_data_z, scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], statistic='mean', bins=z_bins)
    centers_data = (edges_data[1:] + edges_data[:-1]) / 2
    plt.plot(centers_data, stat_data, label="Data All Mean", color="red")
    std_data, edges_data, _ = binned_statistic(all_data_z, scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)["mB"], statistic='std', bins=z_bins)
    #plt.fill_between(centers_data, stat_data - std_data, stat_data + std_data, color="red", alpha=0.2)

    #plt.scatter(scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)["zHD"], scone_cut(df_sim_all, scone_cut_thresh = scone_cut_thresh)["mB"], alpha=0.5, label="Sim All")
    #plt.scatter(scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["zHD"], scone_cut(df_data, scone_cut_thresh = scone_cut_thresh)["mB"], alpha=0.5, label="Data All")
    plt.legend()
    #plt.yscale("log")
    plt.xlabel("Z")
    plt.ylabel("mB")


def sim_genz_plot(df_sim_cc, df_sim_ia, column = 0, survey = "", ):
    plt.subplot(1, 2, column + 1)
    plt.title(f"{survey} Sim Gen Z")

    if "scone" in survey.lower():
        histtype = "step"
    else:
        histtype = "bar"

    try:
        plt.hist(df_sim_cc["SIM_ZCMB"], alpha=0.5, label="Sim CC", histtype=histtype)
        plt.hist(df_sim_ia["SIM_ZCMB"], alpha=0.5, label="Sim IA", histtype=histtype)
        plt.legend()
        plt.yscale("log")
    except:
        plt.hist(df_sim_cc["GENZ"], alpha=0.5, label="Sim CC", histtype=histtype)
        plt.hist(df_sim_ia["GENZ"], alpha=0.5, label="Sim IA", histtype=histtype)
        plt.legend()
        plt.yscale("log")

def percentiles_as_a_function_of_z(df, col, redshift_col, redshift_bins, label = ""):
    from scipy.stats import binned_statistic

    data = scone_cut(df, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    redshift_data = scone_cut(df, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[redshift_col]

    median = binned_statistic(redshift_data, data, statistic='median', bins=redshift_bins)[0]
    p84 = binned_statistic(redshift_data, data, statistic=lambda x: np.percentile(x, 84), bins=redshift_bins)[0]
    p16 = binned_statistic(redshift_data, data, statistic=lambda x: np.percentile(x, 16), bins=redshift_bins)[0]

    bin_centers = redshift_bins[:-1] + np.diff(redshift_bins) / 2
    plt.plot(bin_centers, median, label=f'Median {col}' + label + "PIa > " + str(scone_cut_thresh))
    plt.fill_between(bin_centers, p16, p84, alpha=0.2)
    plt.legend()



#sdss_base_data_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"
#sdss_base_path_sim_all = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-JH8_SDSS-0001/FITOPT000.FITRES.gz"

#sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-SMALL_SDSS-0001/FITOPT000.FITRES.gz"

#sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL-NEWSCONE/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-SMALL-NEWSCONE_SDSS-0001/FITOPT000.FITRES.gz"
#sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-SMALL-NEWSCONE/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"


# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-TBIN/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-SALTCHANGE_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-TBIN/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"


# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-EXTRAPFLUX-REDO/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-EXTRAPFLUX_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-EXTRAPFLUX-REDO/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"



# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NBINT-FIX/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-EXTRAPFLUX_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NBINT-FIX/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"


# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NOUBAND-TEST/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-NOUBAND-TEST_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NOUBAND-TEST/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"


# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NEWSCONE_NOUBAND/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-NEWSCONE_NOUBAND_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NEWSCONE_NOUBAND/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"

#sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NEWTRESTSCONE/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-NEWTRESTSCONE_SDSS-0001/FITOPT000.FITRES.gz"
#sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NEWTRESTSCONE/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS/FITOPT000.FITRES.gz"

# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-FIXPKMJD/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-FIXPKMJD_SDSS-0001/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-FIXPKMJD/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS_PK20/FITOPT000.FITRES.gz"

# sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-POSTSCONEFIX/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-POSTSCONEFIX_SDSS-0002/FITOPT000.FITRES.gz"
# sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-POSTSCONEFIX/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS_PK20/FITOPT000.FITRES.gz"


sdss_base_path_sim_all = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-1016MASK_P22FIT/5_MERGE/MERGE_SDSSFIT_SDSS/output/PIP_CFM-SDSS-1016MASK_P22FIT_SDSS-0001/FITOPT000.FITRES.gz"
sdss_base_data_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-1016MASK_P22FIT/5_MERGE/MERGE_SDSSFIT_DATASDSS/output/SDSS_allCandidates+BOSS_PK20MSK/FITOPT000.FITRES.gz"

sdss_df_sim_all = pd.read_csv(sdss_base_path_sim_all, sep = r"\s+", comment="#")
sdss_df_sim_ia = sdss_df_sim_all[np.isin(sdss_df_sim_all["SIM_GENTYPE"], [1])]
sdss_df_sim_cc = sdss_df_sim_all[~np.isin(sdss_df_sim_all["SIM_GENTYPE"], [1])]
sdss_df_data = pd.read_csv(sdss_base_data_path, sep = r"\s+", comment="#")

if load_dump:

    #sdss_dump_all_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/CFM-SDSS-JH8/1_SIM/SDSS_BCOR/PIP_CFM-SDSS-JH8_SDSS_BCOR/PIP_CFM-SDSS-JH8_SDSS_BCOR.DUMP.gz"
    sdss_dump_all_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-NOUBAND-TEST/1_SIM/SDSS/PIP_CFM-SDSS-NOUBAND-TEST_SDSS-0001/PIP_CFM-SDSS-NOUBAND-TEST_SDSS-0001.DUMP.gz"
    columns = pd.read_csv(sdss_dump_all_path, sep = r"\s+", comment="#",  nrows=0).columns.tolist()

    sdss_dump_all = pd.read_csv(sdss_dump_all_path, sep = r"\s+", comment="#", usecols = ["GENZ", "GENTYPE"])
    sdss_dump_ia = sdss_dump_all[np.isin(sdss_dump_all["GENTYPE"], [1])]
    sdss_dump_cc = sdss_dump_all[~np.isin(sdss_dump_all["GENTYPE"], [1])]

# des_base_data_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/5_MERGE/MERGE_Dsys_DATADESSMP/output/DES-SN5YR_DES/FITOPT000.FITRES.gz"
# des_base_path_sim = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/5_MERGE/MERGE_Dsys_DATADES_SIM_IA/output/PIP_D5YR_RATEPZ_NEWCC_SIM_NOMINAL_DATADESSIM_IA-0001/FITOPT000.FITRES.gz"

fitopt = 0
des_base_data_path = f"/scratch/midway2/rkessler/PIPPIN_OUTPUT/D5YR_RATEPZ_SDSSSYNC_MINI_G10_PHASECOR/5_MERGE/MERGE_Dsys_DATADESSMP/output/DES-SN5YR_DES/FITOPT00{fitopt}.FITRES.gz"
des_base_path_sim = f"/scratch/midway2/rkessler/PIPPIN_OUTPUT/D5YR_RATEPZ_SDSSSYNC_MINI_G10_PHASECOR/5_MERGE/MERGE_Dsys_DATADESSIM_IA/output/PIP_D5YR_RATEPZ_SDSSSYNC_MINI_G10_PHASECOR_DATADESSIM_IA-0001/FITOPT00{fitopt}.FITRES.gz"

des_base_path_sim_cc = des_base_path_sim.replace("SIM_IA", "SIM_CC")
des_df_sim_ia = pd.read_csv(des_base_path_sim, sep = r"\s+", comment="#")
des_df_sim_cc = pd.read_csv(des_base_path_sim_cc, sep = r"\s+", comment="#")
des_df_data = pd.read_csv(des_base_data_path, sep = r"\s+", comment="#")
des_df_sim_all = pd.concat([des_df_sim_ia, des_df_sim_cc])
des_df_sim_all["PROB_SCONE"] = pd.concat([des_df_sim_ia[get_scone_col(des_df_sim_ia)], des_df_sim_cc[get_scone_col(des_df_sim_cc)]])
# delete other prob scone columns
des_df_sim_all = des_df_sim_all.drop(columns=[c for c in des_df_sim_all.columns if "SCONE" in c and c != "PROB_SCONE"])

if load_dump:
    des_dump_cc_path = "/project2/rkessler/SURVEYS/ROMAN/USERS/cmeldorf/D5YR_RATEPZ_NEWCC_ANALYSIS/1_SIM/DESCC_NOMINAL/PIP_D5YR_RATEPZ_NEWCC_BCOR_NOMINAL_DESCC_NOMINAL/PIP_D5YR_RATEPZ_NEWCC_BCOR_NOMINAL_DESCC_NOMINAL.DUMP.gz"
    des_dump_cc = pd.read_csv(des_dump_cc_path, sep = r"\s+", comment="#", usecols = ["GENZ", "GENTYPE"])
    des_dump_ia_path = des_dump_cc_path.replace("DESCC", "DESIA")
    des_dump_ia = pd.read_csv(des_dump_ia_path, sep = r"\s+", comment="#", usecols = ["GENZ", "GENTYPE"])
    des_dump_all = pd.concat([des_dump_ia, des_dump_cc])

all_dfs_des = [des_df_sim_all, des_df_sim_ia, des_df_sim_cc, des_df_data]
all_dfs_sdss = [sdss_df_sim_all, sdss_df_sim_ia, sdss_df_sim_cc, sdss_df_data]


def one_thing_plot(thing):
    plt.figure(figsize = (12, 6), dpi = 200)
    simple_histogram_plot(all_dfs_des[0], all_dfs_des[3], thing, survey = "DES", plot_column = 0)
    simple_histogram_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing, survey = "SDSS", plot_column = 1)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(f"plotting_scripts/{thing}_comparison.png")
    plt.close()
    print(f"{thing} comparison plots saved as '{thing}_comparison.png'")

def one_thing_plot_cc_rescaled(thing = "PROB_SCONE"):
    plt.figure(figsize = (12, 6), dpi = 200)
    cc_rescaled_histogram_plot(all_dfs_des[0], all_dfs_des[3], thing, survey = "DES", plot_column = 0)
    cc_rescaled_histogram_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing, survey = "SDSS", plot_column = 1)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(f"plotting_scripts/{thing}_comparison_CCrescaled.png")
    plt.close()
    print(f"{thing} CC-rescaled comparison plots saved as '{thing}_comparison_CCrescaled.png'")

def cut_data_and_add_cols(df_list, z_high, z_low, zcol="zHD"):
    for i, df in enumerate(df_list):
        df_list[i] = df[df[zcol] < z_high]
        df_list[i] = df_list[i][df_list[i][zcol] > z_low]
        df_list[i] = apply_lc_cuts(df_list[i])
        df_list[i]["HUBRES"] = hubble_residual(df_list[i])
        df_list[i]["PROB_SCONE"] = df_list[i][get_scone_col(df_list[i])]

    return df_list

all_dfs_des = cut_data_and_add_cols(all_dfs_des, 0.9, 0.1, zcol="zHD")
all_dfs_sdss = cut_data_and_add_cols(all_dfs_sdss, 0.35, 0.05, zcol="zHD")
all_dfs = all_dfs_des + all_dfs_sdss
# check that every df has PROB_SCONE column
for i, df in enumerate(all_dfs_sdss):
    if "PROB_SCONE" not in df.columns:
        raise ValueError(f"Missing PROB_SCONE in df {i}")

print("Size of SDSS sim all:", all_dfs_sdss[0].shape)
print("Size of SDSS sim ia:", all_dfs_sdss[1].shape)
print("Size of SDSS sim cc:", all_dfs_sdss[2].shape)
print("Size of SDSS data:", all_dfs_sdss[3].shape)

one_thing_plot("RA")

thing1 = "RA"
thing2 = "MWEBV"
plt.figure(figsize = (12, 6), dpi = 200)
blank_versus_blank_scatter_plot(all_dfs_des[0], all_dfs_des[3], thing1, thing2, survey = "DES", plot_column = 0)
blank_versus_blank_scatter_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing1, thing2, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing1}_versus_{thing2}_comparison.png")
plt.close()
print(f"{thing1} versus {thing2} comparison plots saved as '{thing1}_versus_{thing2}_comparison.png'")



plt.figure(figsize = (12, 6), dpi = 200)
SNRMAX_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
SNRMAX_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/SNRMAX_comparison.png")
plt.close()
print("SNRMAX comparison plots saved as 'SNRMAX_comparison.png'")




plt.figure(figsize = (12, 6), dpi = 200)
mB_as_a_function_of_z(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
mB_as_a_function_of_z(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/mB_as_a_function_of_z_comparison.png")
plt.close()
print("mB as a function of z comparison plots saved as 'mB_as_a_function_of_z_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
mb_plot_split_on_type(all_dfs_des[2], all_dfs_des[1], all_dfs_des[3], survey = "DES", column = 0)
mb_plot_split_on_type(all_dfs_sdss[2], all_dfs_sdss[1], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/mB_comparison_split_on_type.png")
plt.close()
print("mB comparison plots saved as 'mB_comparison_split_on_type.png'")


one_thing_plot("PROB_SCONE")
one_thing_plot_cc_rescaled("PROB_SCONE")
one_thing_plot("mB")
one_thing_plot("x1")
one_thing_plot("c")
one_thing_plot("FITPROB")
one_thing_plot("mBERR")
one_thing_plot("HUBRES")
one_thing_plot("FITCHI2")
one_thing_plot("RA")
one_thing_plot("DEC")
one_thing_plot("PKMJD")


# ---------------------------------------------------------------------------
# SDSS: average histograms over the sim realizations (..._SDSS-0001 ... _SDSS-0025)
# ---------------------------------------------------------------------------
do_sdss_realization_average = True
n_sdss_realizations = 25
# "spread" band shown on the plot: "std" = scatter of a single realization,
# "sem" = uncertainty on the mean (std / sqrt(N)).
realization_err = "std"
sdss_sim_path_template = sdss_base_path_sim_all.replace("_SDSS-0002", "_SDSS-{:04d}")


def load_sim_realizations(path_template, n_realizations, z_high, z_low, zcol = "zHD"):
    dfs = []
    for i in range(1, n_realizations + 1):
        path = path_template.format(i)
        try:
            df = pd.read_csv(path, sep = r"\s+", comment = "#")
        except FileNotFoundError:
            print(f"WARNING: missing realization {i:04d} ({path}), skipping")
            continue
        df = cut_data_and_add_cols([df], z_high, z_low, zcol = zcol)[0]
        dfs.append(df)
    print(f"Loaded {len(dfs)} / {n_realizations} sim realizations")
    return dfs


def _realization_hist_bins(col, real_data, sim_data_list):
    if col == "c":
        return np.linspace(-0.5, 0.5, 20)
    if col == "x1":
        return np.linspace(-3, 3, 20)
    if "scone" in col.lower():
        if inverse:
            return np.linspace(0, scone_cut_thresh, 10)
        n = 20 if scone_cut_thresh <= 0.1 else 10
        return np.linspace(scone_cut_thresh, 1, n)
    # common edges for every realization + data so the counts can be averaged
    return np.histogram_bin_edges(pd.concat([real_data] + list(sim_data_list)), bins = 20)


def realization_averaged_histogram_plot(sim_dfs, df_data, col, survey = "", plot_column = 0, err = "std"):
    """Mean (over sim realizations) histogram of All / Ia / CC, each realization normalized
    to the number of data events, with a band showing the scatter across realizations."""
    plt.subplot(1, 2, plot_column + 1)
    sign = ">" if not inverse else "<="

    real_data = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    sim_cut = [scone_cut(df, scone_cut_thresh = scone_cut_thresh, inverse = inverse) for df in sim_dfs]
    bins = _realization_hist_bins(col, real_data, [d[col] for d in sim_cut])

    n_data, _ = np.histogram(real_data, bins = bins)

    counts = {"All": [], "Ia": [], "CC": []}
    chi_list = []
    for d in sim_cut:
        w = real_data.shape[0] / d.shape[0]
        is_ia = np.isin(d["SIM_GENTYPE"], [1])
        n_all_raw = np.histogram(d[col], bins = bins)[0]
        counts["All"].append(w * n_all_raw)
        counts["Ia"].append(w * np.histogram(d[col][is_ia], bins = bins)[0])
        counts["CC"].append(w * np.histogram(d[col][~is_ia], bins = bins)[0])
        # chi^2 for this realization: data Poisson + this realization's (scaled) sim Poisson
        var = n_data + w**2 * n_all_raw
        var = np.where(var == 0, 1, var)
        chi_list.append(np.nansum((n_data - w * n_all_raw)**2 / var))

    n_real = len(sim_cut)
    stats = {}
    for label, color in [("All", "C0"), ("Ia", "C1"), ("CC", "C2")]:
        arr = np.array(counts[label])
        mean = arr.mean(axis = 0)
        std = arr.std(axis = 0, ddof = 1) if n_real > 1 else np.zeros_like(mean)
        stats[label] = (mean, std)
        band = std / np.sqrt(n_real) if err == "sem" else std
        lo = np.clip(mean - band, 1e-3, None)
        hi = mean + band
        plt.stairs(mean, bins, label = f"Sim {label} (mean of {n_real})", color = color)
        plt.fill_between(bins, np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                         step = "post", alpha = 0.25, color = color, lw = 0)

    plt.errorbar((bins[:-1] + bins[1:]) / 2, n_data, yerr = np.sqrt(n_data), fmt = 'o',
                 label = "Data All", color = "black")

    # reported chi^2 = mean over realizations of each realization's chi^2
    chi_arr = np.array(chi_list)
    chisq = chi_arr.mean()
    chisq_std = chi_arr.std(ddof = 1) if n_real > 1 else 0.0
    dof = len(n_data)

    plt.title(f"{survey} PIa {sign} {scone_cut_thresh}, {n_real} realizations\n"
              f"<Chi^2> = {chisq:.3f} +/- {chisq_std:.3f}, dof = {dof}")
    n_ia_mean = np.mean([np.isin(d["SIM_GENTYPE"], [1]).sum() for d in sim_cut])
    n_cc_mean = np.mean([(~np.isin(d["SIM_GENTYPE"], [1])).sum() for d in sim_cut])
    plt.text(0.95, 0.95, f"<Sim Ia>: {n_ia_mean:.0f}, <Sim CC>: {n_cc_mean:.0f}\nband = {err}",
             transform = plt.gca().transAxes, verticalalignment = 'top', horizontalalignment = 'right')
    print(f"[realization avg] {survey} {col}: per-realization chi^2 = "
          + ", ".join(f"{c:.2f}" for c in chi_arr))
    print(f"[realization avg] Mean reduced Chi^2 for {survey} {col}: {chisq/dof:.3f}")

    plt.legend(loc = "lower left", fontsize = 8)
    plt.yscale("log")
    plt.xlabel(col)
    plt.ylabel("Counts")
    plt.ylim(0.1, None)


def one_thing_plot_realization_averaged(thing, sdss_sim_dfs):
    plt.figure(figsize = (12, 6), dpi = 200)
    # DES only has a single sim, so keep the standard plot there
    simple_histogram_plot(all_dfs_des[0], all_dfs_des[3], thing, survey = "DES", plot_column = 0)
    realization_averaged_histogram_plot(sdss_sim_dfs, all_dfs_sdss[3], thing, survey = "SDSS",
                                        plot_column = 1, err = realization_err)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    outname = f"plotting_scripts/{thing}_comparison_SDSSavg{len(sdss_sim_dfs)}.png"
    plt.savefig(outname)
    plt.close()
    print(f"{thing} realization-averaged comparison saved as '{outname}'")


def realization_averaged_cc_rescaled_plot(sim_dfs, df_data, col = "PROB_SCONE", survey = "", plot_column = 0,
                                          err = "std"):
    """CC-rescaled histogram averaged over sim realizations. The CC scale factor s is fit
    independently for each realization; the plot shows the mean Ia / CC / Ia+CC over
    realizations with a band showing the scatter across realizations."""
    plt.subplot(1, 2, plot_column + 1)
    sign = ">" if not inverse else "<="

    real_data = scone_cut(df_data, scone_cut_thresh = scone_cut_thresh, inverse = inverse)[col]
    sim_cut = [scone_cut(df, scone_cut_thresh = scone_cut_thresh, inverse = inverse) for df in sim_dfs]

    if col == "c":
        bins = np.linspace(-0.5, 0.5, 20)
    elif col == "x1":
        bins = np.linspace(-3, 3, 20)
    else:
        # common edges across realizations so the counts can be averaged
        bins = np.histogram_bin_edges(pd.concat([d[col] for d in sim_cut]), bins = 10)

    n_data, _ = np.histogram(real_data, bins = bins)

    ia_list, cc_list, tot_list, s_list, chi_list, chi1_list = [], [], [], [], [], []
    for d in sim_cut:
        is_ia = np.isin(d["SIM_GENTYPE"], [1])
        n_ia, _ = np.histogram(d[col][is_ia], bins = bins)
        n_cc, _ = np.histogram(d[col][~is_ia], bins = bins)
        w = real_data.shape[0] / d.shape[0]
        s_i, chi_i, chi1_i = fit_cc_scale(n_ia, n_cc, n_data, w)
        ia_list.append(w * n_ia)
        cc_list.append(w * s_i * n_cc)
        tot_list.append(w * n_ia + w * s_i * n_cc)
        s_list.append(s_i)
        chi_list.append(chi_i)
        chi1_list.append(chi1_i)

    n_real = len(sim_cut)
    s_arr = np.array(s_list)
    s_mean = s_arr.mean()
    s_std = s_arr.std(ddof = 1) if n_real > 1 else 0.0

    stats = {}
    for label, arr_list, color, lw in [("Ia + CC", tot_list, "C0", 2), ("Ia", ia_list, "C1", 1),
                                       ("CC (rescaled)", cc_list, "C2", 1)]:
        arr = np.array(arr_list)
        mean = arr.mean(axis = 0)
        std = arr.std(axis = 0, ddof = 1) if n_real > 1 else np.zeros_like(mean)
        stats[label] = (mean, std)
        band = std / np.sqrt(n_real) if err == "sem" else std
        lo = np.clip(mean - band, 1e-3, None)
        hi = mean + band
        plt.stairs(mean, bins, label = f"Sim {label} (mean of {n_real})", color = color, lw = lw)
        plt.fill_between(bins, np.append(lo, lo[-1]), np.append(hi, hi[-1]),
                         step = "post", alpha = 0.25, color = color, lw = 0)

    plt.errorbar((bins[:-1] + bins[1:]) / 2, n_data, yerr = np.sqrt(n_data), fmt = 'o',
                 label = "Data All", color = "black")

    # reported chi^2 = mean over realizations of each realization's best-fit chi^2
    chi_arr = np.array(chi_list)
    chisq = chi_arr.mean()
    chisq_std = chi_arr.std(ddof = 1) if n_real > 1 else 0.0
    dof = len(n_data)

    plt.title(f"{survey} PIa {sign} {scone_cut_thresh}, {n_real} realizations\n"
              f"<Chi^2> = {chisq:.2f} +/- {chisq_std:.2f} (s=1: {np.mean(chi1_list):.2f}), dof = {dof}")
    plt.text(0.95, 0.95, f"CC rescale = {s_mean:.3f} +/- {s_std:.3f}\nband = {err}",
             transform = plt.gca().transAxes, verticalalignment = 'top', horizontalalignment = 'right')

    print(f"[realization avg] {survey} {col}: per-realization CC rescale = "
          + ", ".join(f"{s:.3f}" for s in s_arr))
    print(f"[realization avg] {survey} {col}: CC rescale = {s_mean:.3f} +/- {s_std:.3f}, "
          f"<chi^2> = {chisq:.3f} +/- {chisq_std:.3f}, <chi^2(s=1)> = {np.mean(chi1_list):.3f}, dof = {dof}")
    print(f"[realization avg] {survey} {col}: per-realization chi^2 = "
          + ", ".join(f"{c:.2f}" for c in chi_arr))

    plt.legend(loc = "upper left", fontsize = 8)
    plt.yscale("log")
    plt.xlabel(col)
    plt.ylabel("Counts")
    plt.ylim(0.1, 2e3)
    return s_arr


def one_thing_plot_cc_rescaled_realization_averaged(sdss_sim_dfs, thing = "PROB_SCONE"):
    plt.figure(figsize = (12, 6), dpi = 200)
    # DES only has a single sim, so keep the standard CC-rescaled plot there
    cc_rescaled_histogram_plot(all_dfs_des[0], all_dfs_des[3], thing, survey = "DES", plot_column = 0)
    realization_averaged_cc_rescaled_plot(sdss_sim_dfs, all_dfs_sdss[3], thing, survey = "SDSS",
                                          plot_column = 1, err = realization_err)
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    outname = f"plotting_scripts/{thing}_comparison_CCrescaled_SDSSavg{len(sdss_sim_dfs)}.png"
    plt.savefig(outname)
    plt.close()
    print(f"{thing} CC-rescaled realization-averaged comparison saved as '{outname}'")


if do_sdss_realization_average:
    sdss_sim_realizations = load_sim_realizations(sdss_sim_path_template, n_sdss_realizations,
                                                  z_high = 0.35, z_low = 0.05, zcol = "zHD")
    for thing in ["PROB_SCONE", "mB", "x1", "c", "FITPROB", "mBERR", "HUBRES", "FITCHI2", "zHD"]:
        one_thing_plot_realization_averaged(thing, sdss_sim_realizations)
    one_thing_plot_cc_rescaled_realization_averaged(sdss_sim_realizations, "PROB_SCONE")


plt.subplot(1,2,1)
percentiles_as_a_function_of_z(all_dfs_des[0], "x1", "zHD", np.arange(0.1, 0.9, 0.05), label = f" DES Sim PIa > {scone_cut_thresh}")
percentiles_as_a_function_of_z(all_dfs_des[3], "x1", "zHD", np.arange(0.1, 0.9, 0.05), label =f" DES Data PIa > {scone_cut_thresh}")
plt.subplot(1,2,2)
percentiles_as_a_function_of_z(all_dfs_sdss[0], "x1", "zHD", np.arange(0.05, 0.35, 0.05), label = f" SDSS Sim PIa > {scone_cut_thresh}")
percentiles_as_a_function_of_z(all_dfs_sdss[3], "x1", "zHD", np.arange(0.05, 0.35, 0.05), label = f" SDSS Data PIa > {scone_cut_thresh}")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.legend()
plt.savefig(f"plotting_scripts/x1_percentiles_as_a_function_of_z_comparison.png")
plt.close()
print(f"x1 percentiles as a function of z comparison plots saved as 'x1_percentiles_as_a_function_of_z_comparison.png'")


thing = "FITPROB"
plt.subplot(1,2,1)
percentiles_as_a_function_of_z(all_dfs_des[0], thing, "zHD", np.arange(0.1, 0.9, 0.1), label = f" DES Sim PIa > {scone_cut_thresh}")
percentiles_as_a_function_of_z(all_dfs_des[3], thing, "zHD", np.arange(0.1, 0.9, 0.1), label =f" DES Data PIa > {scone_cut_thresh}")
plt.subplot(1,2,2)
percentiles_as_a_function_of_z(all_dfs_sdss[0], thing, "zHD", np.arange(0.05, 0.4, 0.05), label = f" SDSS Sim PIa > {scone_cut_thresh}")
percentiles_as_a_function_of_z(all_dfs_sdss[3], thing, "zHD", np.arange(0.05, 0.4, 0.05), label = f" SDSS Data PIa > {scone_cut_thresh}")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.legend()
plt.savefig(f"plotting_scripts/{thing}_percentiles_as_a_function_of_z_comparison.png")
plt.close()
print(f"{thing} percentiles as a function of z comparison plots saved as '{thing}_percentiles_as_a_function_of_z_comparison.png'")


thing = "FITPROB"
plt.figure(figsize = (12, 6), dpi = 200)
simple_histogram_plot(all_dfs_des[0], all_dfs_des[3], thing, survey = "DES", plot_column = 0)
simple_histogram_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing}_comparison.png")
plt.close()
print(f"{thing} comparison plots saved as '{thing}_comparison.png'")

plt.figure(figsize = (12, 6), dpi = 200)
hubble_residual_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
hubble_residual_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/hubble_residual_comparison.png")
plt.close()
print("Hubble residual comparison plots saved as 'hubble_residual_comparison.png'")

thing1 = "x1"
thing2 = "SNRMAX1"
plt.figure(figsize = (12, 6), dpi = 200)
blank_versus_blank_scatter_plot(all_dfs_des[0], all_dfs_des[3], thing1, thing2, survey = "DES", plot_column = 0)
blank_versus_blank_scatter_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing1, thing2, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing1}_versus_{thing2}_comparison.png")
plt.close()
print(f"{thing1} versus {thing2} comparison plots saved as '{thing1}_versus_{thing2}_comparison.png'")


thing1 = "zHD"
thing2 = "FITPROB"
thing3 = "x1"
plt.figure(figsize = (12, 6), dpi = 200)
blank_versus_blank_scatter_plot(all_dfs_des[0], all_dfs_des[3], thing1, thing2, col3=thing3, survey = "DES", plot_column = 0)
blank_versus_blank_scatter_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing1, thing2, col3=thing3, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing1}_versus_{thing2}_comparison.png")
plt.close()
print(f"{thing1} versus {thing2} comparison plots saved as '{thing1}_versus_{thing2}_comparison.png'")


thing1 = "zHD"
thing2 = "SNRMAX1"
plt.figure(figsize = (12, 6), dpi = 200)
blank_versus_blank_scatter_plot(all_dfs_des[0], all_dfs_des[3], thing1, thing2, survey = "DES", plot_column = 0)
blank_versus_blank_scatter_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing1, thing2, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing1}_versus_{thing2}_comparison.png")
plt.close()
print(f"{thing1} versus {thing2} comparison plots saved as '{thing1}_versus_{thing2}_comparison.png'")



thing1 = "x1"
thing2 = "x1ERR"
plt.figure(figsize = (12, 6), dpi = 200)
blank_versus_blank_scatter_plot(all_dfs_des[0], all_dfs_des[3], thing1, thing2, survey = "DES", plot_column = 0)
blank_versus_blank_scatter_plot(all_dfs_sdss[0], all_dfs_sdss[3], thing1, thing2, survey = "SDSS", plot_column = 1)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig(f"plotting_scripts/{thing1}_versus_{thing2}_comparison.png")
plt.close()
print(f"{thing1} versus {thing2} comparison plots saved as '{thing1}_versus_{thing2}_comparison.png'")


data_versus_data_histogram_plot(all_dfs_des[0], "DES", all_dfs_sdss[0], "SDSS", col = "FITPROB", label = "SIM")
plt.savefig("plotting_scripts/FITPROB_sim_comparison.png")
plt.close()

data_versus_data_histogram_plot(all_dfs_des[3], "DES", all_dfs_sdss[3], "SDSS", col = "FITPROB", label = "DATA")
plt.savefig("plotting_scripts/FITPROB_data_comparison.png")
plt.close()



# plt.figure(figsize = (12, 6), dpi = 200)
# mb_err_plot(all_dfs_des[0], all_dfs_des[3], survey = "DES", column = 0)
# mb_err_plot(all_dfs_sdss[0], all_dfs_sdss[3], survey = "SDSS", column = 1)
# plt.subplots_adjust(hspace=0.4, wspace=0.4)
# plt.savefig("plotting_scripts/mBERR_comparison.png")
# plt.close()
# print("mBERR comparison plots saved as 'mBERR_comparison.png'")


plt.figure(figsize = (12, 6), dpi = 200)
sim_genz_plot(all_dfs_des[2], all_dfs_des[1], column = 0, survey = "DES")
sim_genz_plot(all_dfs_sdss[2], all_dfs_sdss[1], column = 1, survey = "SDSS")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/dump_genz_comparison.png")
plt.close()
print("Sim GENZ comparison plots saved as 'dump_genz_comparison.png'")



scone_SDSS_df_path = "/scratch/midway2/rkessler/PIPPIN_OUTPUT/CFM-SDSS-RATE/1_SIM/SDSS_BCOR/PIP_CFM-SDSS-RATE_SDSS_BCOR/PIP_CFM-SDSS-RATE_SDSS_BCOR.DUMP.gz"
scone_SDSS_df = pd.read_csv(scone_SDSS_df_path, comment = "#", sep=r"\s+", usecols=["GENZ", "GENTYPE"])

scone_SDSS_df = scone_SDSS_df[scone_SDSS_df["GENZ"] > 0.05]
scone_SDSS_df = scone_SDSS_df[scone_SDSS_df["GENZ"] < 0.35]
scone_SDSS_df_cc = scone_SDSS_df[~np.isin(scone_SDSS_df["GENTYPE"], [1])]
scone_SDSS_df_ia = scone_SDSS_df[np.isin(scone_SDSS_df["GENTYPE"], [1])]

if load_dump:
    des_dump_cc = des_dump_cc[des_dump_cc["GENZ"] > 0.1]
    des_dump_cc = des_dump_cc[des_dump_cc["GENZ"] < 0.9]
    des_dump_ia = des_dump_ia[des_dump_ia["GENZ"] > 0.1]
    des_dump_ia = des_dump_ia[des_dump_ia["GENZ"] < 0.9]

    sdss_dump_cc = sdss_dump_cc[sdss_dump_cc["GENZ"] > 0.05]
    sdss_dump_cc = sdss_dump_cc[sdss_dump_cc["GENZ"] < 0.35]
    sdss_dump_ia = sdss_dump_ia[sdss_dump_ia["GENZ"] > 0.05]
    sdss_dump_ia = sdss_dump_ia[sdss_dump_ia["GENZ"] < 0.35]

    plt.figure(figsize = (12, 6), dpi = 200)
    sim_genz_plot(des_dump_cc, des_dump_ia, column = 0, survey = "DES")
    sim_genz_plot(sdss_dump_cc, sdss_dump_ia, column = 1, survey = "SDSS")
    print("Total size sdss_dump_cc:", sdss_dump_cc.shape)
    print("Total size sdss_dump_ia:", sdss_dump_ia.shape)
    print("Total size scone_SDSS_df_cc:", scone_SDSS_df_cc.shape)
    print("Total size scone_SDSS_df_ia:", scone_SDSS_df_ia.shape)
    sim_genz_plot(scone_SDSS_df_cc, scone_SDSS_df_ia, column = 1, survey = "SDSS scone trainset")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig("plotting_scripts/sim_genz_comparison.png")
    plt.close()
    print("Sim GENZ comparison plots saved as 'sim_genz_comparison.png'")



def sim_genz_plot_by_type(dumpall1, dumpall2, survey1, survey2, fitdf1=None, fitdf2=None):
    types = np.unique(np.concatenate([dumpall1["GENTYPE"].values, dumpall2["GENTYPE"].values]))
    plt.figure(figsize=(24, 6))
    plt.subplots(len(types), 1)
    bins = np.linspace(0, 1, 50)
    data1 = dumpall1["GENZ"]
    data2 = dumpall2["GENZ"]
    counts1, _ = np.histogram(data1, bins=bins)
    counts2, _ = np.histogram(data2, bins=bins)
    print("counts1[0]:", counts1[10])
    print("counts2[0]:", counts2[10])
    weighting = counts1[10] / counts2[10]
    weighting = 1


    for t in types:
        data1 = dumpall1[dumpall1["GENTYPE"] == t]["GENZ"]
        data2 = dumpall2[dumpall2["GENTYPE"] == t]["GENZ"]

        for c in fitdf1.columns:
            print(c)


        lc_data1 = fitdf1["SIM_ZCMB"][fitdf1["SIM_GENTYPE"] == t] if fitdf1 is not None else None
        lc_data2 = fitdf2["SIM_ZCMB"][fitdf2["SIM_GENTYPE"] == t] if fitdf2 is not None else None
        weights = np.full_like(data2, weighting)

        #weights = np.full_like(data2, np.size(data1)/np.size(data2))
        plt.subplot(len(types), 1, np.where(types == t)[0][0] + 1)
        plt.title(f"GENTYPE {t}")
        plt.hist(data1, alpha=1, label=survey1, histtype="step", bins = bins)
        plt.hist(data2, alpha=1, label=survey2, histtype="step", bins = bins, weights=weights)
        if lc_data1 is not None and lc_data2 is not None:
            plt.hist(lc_data1, alpha=0.5, label=f"{survey1} fit", histtype="step", bins=bins)
            weights = np.full_like(lc_data2, weighting)
            plt.hist(lc_data2, alpha=0.5, label=f"{survey2} fit", histtype="step", bins=bins, weights=weights)
            plt.text(0.95, 0.95, f"{survey1}: total: {len(data1)}, fit: {len(lc_data1)}\n{survey2}: total: {len(data2)}, fit: {len(lc_data2)}",
                     transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

        plt.legend()
        plt.yscale("log")

if load_dump:
    plt.close()
    sim_genz_plot_by_type(sdss_dump_all, des_dump_all, "SDSS", "DES", fitdf1=all_dfs_sdss[0], fitdf2=all_dfs_des[0])
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.savefig("plotting_scripts/sim_genz_comparison_by_type.png")
    print("Saved plot to plotting_scripts/sim_genz_comparison_by_type.png")


def plot_bands_prescence_as_function_of_redshift(redshift_bins, redshift_column, df, bands = ["u", "g", "r", "i", "z"],
ls = "-", color = None, survey = ""):

    for band in bands:
        band_df = df["BANDLIST"].str.contains(band)
        if band_df.sum() == 0:
            print(f"No data for band {band}")
            continue
        else:
            band_df = df[band_df]
        counts, _ = np.histogram(band_df[redshift_column], bins=redshift_bins)
        counts = counts.astype(float)  # Convert counts to float to allow NaN assignment
        counts[np.where(counts == 0)] = np.nan
        total_counts, _ = np.histogram(df[redshift_column], bins=redshift_bins)
        counts = counts / total_counts  # Normalize by total counts
        error = np.sqrt(counts) / total_counts  # Propagate error for normalized counts
        bin_centers = (redshift_bins[:-1] + redshift_bins[1:]) / 2
        plt.errorbar(bin_centers, counts, yerr=error, label=band, ls = ls, color = color)
        plt.title(f"{survey} {band} band")
        plt.legend()


redshift_bins = np.linspace(0, 1, 10)

plt.figure(figsize=(20, 20))
color_dict = {"g": "green", "r": "red", "i": "orange", "z": "purple"}
df_data = all_dfs_des[3]
df_sim = all_dfs_des[0]
for i, b in enumerate(["g", "r", "i", "z"]):
    plt.subplot(4, 2, i + 1)
    plot_bands_prescence_as_function_of_redshift(redshift_bins, "zHD", df_data, survey = "DES", bands = [b], ls = "-", color = color_dict[b])
    plot_bands_prescence_as_function_of_redshift(redshift_bins, "zHD", df_sim, survey = "DES", bands = [b], ls = "--", color = color_dict[b])
df_data = all_dfs_sdss[3]
df_sim = all_dfs_sdss[0]
redshift_bins = np.linspace(0, 0.5, 10)
for i, b in enumerate(["g", "r", "i", "z"]):
    plt.subplot(4, 2, i + 5)
    plot_bands_prescence_as_function_of_redshift(redshift_bins, "zHD", df_data, survey = "SDSS", bands = [b], ls = "-", color = color_dict[b])
    plot_bands_prescence_as_function_of_redshift(redshift_bins, "zHD", df_sim, survey = "SDSS", bands = [b], ls = "--", color = color_dict[b])

plt.suptitle("Fraction of Candidates with Band as Function of Redshift, \n Dashed = Simulated, Solid = Real Data")
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("plotting_scripts/band_fraction_as_function_of_redshift.png")
plt.close()
