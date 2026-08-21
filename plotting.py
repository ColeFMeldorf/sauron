# Standard Library
import logging
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl

# Sauron modules
from funcs import chi2
from funcs import default_parameter_name_dictionary


def update_rcParams(key, val):
    if key in rcParams:
        rcParams[key] = val


def LaurenNicePlots():
    update_rcParams("font.size", 10)
    update_rcParams("font.family", "serif")
    update_rcParams("xtick.major.size", 8)
    update_rcParams("xtick.labelsize", "large")
    update_rcParams("xtick.direction", "in")
    update_rcParams("xtick.minor.visible", True)
    update_rcParams("xtick.top", True)
    update_rcParams("ytick.major.size", 8)
    update_rcParams("ytick.labelsize", "large")
    update_rcParams("ytick.direction", "in")
    update_rcParams("ytick.minor.visible", True)
    update_rcParams("ytick.right", True)
    update_rcParams("xtick.minor.size", 4)
    update_rcParams("ytick.minor.size", 4)
    update_rcParams("xtick.major.pad", 10)
    update_rcParams("ytick.major.pad", 10)
    update_rcParams("legend.numpoints", 1)
    update_rcParams("mathtext.fontset", "cm")
    update_rcParams("mathtext.rm", "serif")
    update_rcParams("axes.labelsize", "x-large")
    update_rcParams("lines.marker", "None")
    update_rcParams("lines.markersize", 1)
    update_rcParams("lines.markeredgewidth", 1.0)
    update_rcParams("lines.markeredgecolor", "auto")

    cycle_colors = ["navy", "maroon", "darkorange", "darkorchid",  "6FADFA", "7D7D7D", "black"]
    # cycle_markers = ["o", "^", "*", "s", "X", "d", "1", "2", "3"]
    # cycle_colors = ['darkorchid','darkorange','darkturquoise']
    # cycle_markers = ['o','^','*']
    # + mpl.cycler(marker=cycle_markers)
    update_rcParams("axes.prop_cycle", mpl.cycler(color=cycle_colors) )


def summary_plot(runner):
    """ Generate a plot of the results, including predicted vs observed counts and chi2 contours."""
    surveys = runner.results.keys()
    LaurenNicePlots()

    fig = plt.figure(figsize=(9, 4), dpi=200)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    plt.tight_layout()
    fig.tight_layout(pad=3.0)

    surveys = list(surveys)
    for i, survey in enumerate(surveys):
        s = survey
        if survey != "combined":

            _plot_binned_rate(survey, runner, ax1)

            ax1.set_xlabel("Redshift")
            ax1.set_yticks([2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5])
            ax1.set_yticklabels(["2", "3", "4", "5", "6", "7"])
            ax1.set_ylabel(r"Rate $[\times 10^{-5}$ SNe yr$^{-1}$ Mpc$^{-3}]$")

        if isinstance(runner.results[s], list):
            df = runner.results[s][0]
        else:
            df = runner.results[s]


        if "combined" in surveys:
            extraplot_survey = "combined"
        else:
            extraplot_survey = surveys[-1]

        if survey == extraplot_survey:
            smallest_x = np.inf
            biggest_x = -np.inf
            smallest_y = np.inf
            biggest_y = -np.inf
            things_to_plot = runner.rates_to_plot[survey]
            for ii, item in enumerate(things_to_plot):
                reduced_chi_2 = _plot_one_rate_function_and_error(ii, item, runner, ax1, survey)
                if ii == 0:
                    props = dict(boxstyle="round", facecolor="white", alpha=0.8)

                if item["csfr_name"] is not None:
                    ax1.text(-0.55, 0.9 - ii * 0.1, "Reduced $\\chi^2$ ({} CSFR = {:.2f})".format(csfr_label, reduced_chi_2),
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom",
                            horizontalalignment="right", bbox=props)
                else:
                    ax1.text(-0.85, 0.7 - ii * 0.1, "Reduced $\\chi^2$ = {:.2f}".format(reduced_chi_2),
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment="bottom",
                            horizontalalignment="right", bbox=props)

            ax1.legend(loc = "lower right", fontsize=8, framealpha=0.5)

            if "non_parametric" not in runner.rate_function_name:

                if "combined" in surveys:
                    non_combined_datasets = [survey for survey in surveys if survey != "combined"]
                    label = "+".join(non_combined_datasets)
                else:
                    label = survey


                param_names = runner.param_names
                if param_names is None:
                    param_names = ["param_" + str(i) for i in range(len(runner.final_counts[survey]["result"]))]
                param_names = [p.replace("$", "") for p in param_names]
                param_names = [p.replace("\\", "") for p in param_names]


                df_list = runner.results[s]

                if runner.multiple_csfrs:
                    csfrs = [d["csfr"].values[0] for d in df_list]
                else:
                    csfrs = [None]

                extent_chi_0s = []
                extent_chi_1s = []
                extent_chi_2s = []
                extent_chi_3s = []
                for i, c in enumerate(csfrs):
                    logging.debug(f"Processing CSFR: {c}")
                    df = df_list[i]

                    stretch_x = 5 if c != "L08" else 10
                    stretch_y = 5

                    extent_chi = [df[param_names[1]][0] - stretch_x * df[f"{param_names[1]}_error"][0],
                                    df[param_names[1]][0] + stretch_x * df[f"{param_names[1]}_error"][0],
                                    df[param_names[0]][0] - stretch_y * df[f"{param_names[0]}_error"][0],
                                    df[param_names[0]][0] + stretch_y * df[f"{param_names[0]}_error"][0]]

                    extent_chi_0s.append(extent_chi[0])
                    extent_chi_1s.append(extent_chi[1])
                    extent_chi_2s.append(extent_chi[2])
                    extent_chi_3s.append(extent_chi[3])
                    logging.debug(f"current df {df}")
                    chi2_map = generate_chi2_map(runner, s, extent=extent_chi, index =1, csfr = c) # this needs to be fixed
                    chi2_map -= np.min(chi2_map)

                    sigma_map = chi2_map

                    # Reload param names with Latex included this time.

                    if not runner.multiple_csfrs:
                        # Can't do this with multiple rates from multiple CSFRs being plotted, as which would get to
                        # be the color map? Instead, just plot contours.
                        im = ax2.imshow(sigma_map, extent=extent_chi, origin="lower", aspect="auto", cmap="viridis")
                        plt.colorbar(im, ax=ax2, label="Δχ²")
                    # Δχ² contour levels for 2 parameters (≈1σ, 2σ, 3σ confidence regions; see Numerical Recipes / χ² tables)
                    cs = ax2.contour(sigma_map, levels=[2.30, 6.18, 11.83], extent=extent_chi, colors="k", linewidths=1)
                    contour_level = cs.allsegs[0]
                    segment = contour_level[0]  # This is a NumPy array of shape (N, 2)

                    x_coords = segment[:, 0]
                    min_x = np.min(x_coords)
                    if min_x < smallest_x or smallest_x is None:
                        smallest_x = min_x
                    if min_x > biggest_x or biggest_x is None:
                        biggest_x = min_x
                    y_coords = segment[:, 1]
                    min_y = np.min(y_coords)
                    if min_y < smallest_y or smallest_y is None:
                        smallest_y = min_y
                    if min_y > biggest_y or biggest_y is None:
                        biggest_y = min_y
                    logging.debug(f"Scatter plotting the following values: {df[param_names[1]]}, {df[param_names[0]]}")

                    chi_plot_label = f"{label}"
                    if runner.multiple_csfrs:
                        csfr_label = c
                        csfr_label = csfr_label.replace("$", "").replace("\\", "").replace("_", " ")
                        # Loop through and capitalize first letter of each word
                        for i, letter in enumerate(csfr_label):
                            if letter.isalpha() and (i == 0 or csfr_label[i-1] == " "):
                                csfr_label = csfr_label[:i] + csfr_label[i].upper() + csfr_label[i+1:]
                        chi_plot_label += f" ({csfr_label} CSFR)"
                    chi_plot_label += " (This Work)"
                    ax2.errorbar(df[param_names[1]], df[param_names[0]], xerr=df[f"{param_names[1]}_error"], yerr=df[f"{param_names[0]}_error"], fmt="o",
                                    ms=5, label=chi_plot_label, color = "white")
                _add_literature_values(ax2, runner)

                label_names = default_parameter_name_dictionary.get(runner.rate_function_name, None)
                if label_names is None:
                    label_names = param_names


                ax2.set_xlabel(label_names[1])
                ax2.set_ylabel(label_names[0])

                #ax2.set_yticks([1.9e-5, 2e-5, 2.1e-5, 2.2e-5, 2.3e-5, 2.4e-5, 2.5e-5])
                #ax2.set_yticklabels(["1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5"])
                #ax2.set_ylabel(r"$\alpha [\times 10^{-5}$ SNe yr$^{-1}$ Mpc$^{-3}]$")

                _get_ax2_ticks(ax2, extent_chi_0s, extent_chi_1s, extent_chi_2s, extent_chi_3s)

                ax2.legend(loc = "upper right", fontsize=8, ncol = 2)

    fig.savefig("summary_plot.png")


def _plot_one_rate_function_and_error(ii, item, runner, ax1, survey):
    surveys = runner.results.keys()
    rate_fine = item["predicted_rate"]
    z_centers_fine = item["z_centers_fine"]
    z_centers = item["z_centers"]
    predicted_rate_16 = item["predicted_rate_16"]
    predicted_rate_84 = item["predicted_rate_84"]
    label, csfr_label = _get_label_for_rate(item)
    chi_2 = item["chi"]
    reduced_chi_2 = chi_2 / (len(z_centers) - len(runner.final_counts[survey]["result"]))

    color_index = ii + len(surveys) -1
    color = "C"+str(color_index) if survey != "combined" else "gray"
    if "non_parametric" not in runner.rate_function_name:
        ax1.plot(z_centers_fine, rate_fine, label=label, color = color)
    else:
        ax1.plot(z_centers, rate_fine, label=label, color = color)
    ax1.fill_between(z_centers, predicted_rate_16, predicted_rate_84, color=color, alpha=0.5)
    # , label="1 sigma confidence region"
    return reduced_chi_2


def _plot_binned_rate(survey, runner, ax1):
    z_centers = np.array(runner.fit_args_dict["z_centers"][survey])

    yerr = np.array([runner.final_counts[survey]["binned_rate_84"] -
                    runner.final_counts[survey]["binned_rate"],
                    runner.final_counts[survey]["binned_rate"] - runner.final_counts[survey]["binned_rate_16"]])
    index = -1
    xerr = runner.fit_args_dict["z_centers_err"][survey][index] # this needs to be fixed for multiple sim datasets
    ax1.errorbar(z_centers, runner.final_counts[survey]["binned_rate"], xerr=xerr, yerr=yerr, fmt="o",
                label=f" {survey} Binned Rate ", ms=5)


def _get_label_for_rate(item):
    if item["csfr_name"] is not None:
        csfr_label = item["csfr_name"]
        csfr_label = csfr_label.replace("$", "").replace("\\", "").replace("_", " ")
        # Loop through and capitalize first letter of each word
        for i, letter in enumerate(csfr_label):
            if letter.isalpha() and (i == 0 or csfr_label[i-1] == " "):
                csfr_label = csfr_label[:i] + csfr_label[i].upper() + csfr_label[i+1:]

        label = f"Best Fit Rate ({csfr_label} CSFR)"
    else:
        label = "Best Fit Rate "
        csfr_label = None

    return label, csfr_label


def _get_ax2_ticks(ax2, extent_chi_0s, extent_chi_1s, extent_chi_2s, extent_chi_3s):
    # Adaptively define the ticks
    extent_chi_0s = np.asarray(extent_chi_0s)
    extent_chi_1s = np.asarray(extent_chi_1s)
    extent_chi_2s = np.asarray(extent_chi_2s)
    extent_chi_3s = np.asarray(extent_chi_3s)

    y_limits = ax2.get_ylim()
    y_range = y_limits[1] - y_limits[0]
    y_tick_spacing = y_range / 5  # Aim for around 5 ticks
    y_ticks = np.arange(np.ceil(y_limits[0] / y_tick_spacing) * y_tick_spacing, np.floor(y_limits[1] / y_tick_spacing) * y_tick_spacing + y_tick_spacing, y_tick_spacing)
    ax2.set_yticks(y_ticks)


    log_norm = np.floor(np.log10(np.abs(max(y_ticks))))
    norm = 10**log_norm
    log_norm = int(log_norm)
    # get current y label

    if log_norm < -1 or log_norm > 1:
        current_ylabel = ax2.get_ylabel()
        # update the y label to include the normalization factor
        ax2.set_ylabel(f"{current_ylabel} ["+r"$\times"+"10^"+"{"+str(log_norm)+"}$]")
    else:
        norm = 1

    # Check the labels. If any are non-unique, dial up the precision until they are unique
    precision = 1
    while len(set([f"{y_tick/norm:.{precision}f}" for y_tick in y_ticks])) < len(y_ticks):
        precision += 1

    ax2.set_yticklabels([f"{y_tick/norm:.{precision}f}" for y_tick in y_ticks])

    # # Do the same for x ticks
    # #x_limits = smallest_x, biggest_x
    # x_limits = ax2.get_xlim()
    # x_range = x_limits[1] - x_limits[0]
    # x_tick_spacing = x_range / 5  # Aim for around 5 ticks
    # x_ticks = np.arange(np.ceil(x_limits[0] / x_tick_spacing) * x_tick_spacing, np.floor(x_limits[1] / x_tick_spacing) * x_tick_spacing + x_tick_spacing, x_tick_spacing)
    # #ax2.set_xticks(x_ticks)

    # log_norm = np.floor(np.log10(np.abs(max(x_ticks))))
    # norm = 10**log_norm
    # log_norm = int(log_norm)

    # if "power_law_dtd" in runner.rate_function_name:
    #     x_unit = "SNe yr$^{-1}$ Mpc$^{-3}$"
    # else:
    #     x_unit = ""

    # if log_norm < -1 or log_norm > 1:
    #     # get current x label
    #     current_xlabel = ax2.get_xlabel()
    #     # update the x label to include the normalization factor
    #     ax2.set_xlabel(f"{current_xlabel} ["+r"$\times"+"10^"+"{"+str(log_norm)+"}$]" + f" {x_unit}")
    # else:
    #     norm = 1
    #  # Check the labels. If any are non-unique, dial up the precision until they are unique
    # precision = 1
    # while len(set([f"{x_tick/norm:.{precision}f}" for x_tick in x_ticks])) < len(x_ticks):
    #     precision += 1
    # ax2.set_xticklabels([f"{x_tick/norm:.{precision}f}" for x_tick in x_ticks])


def _two_param_plot_stat_sys(a, a_stat_up, a_stat_low, a_sys_up, a_sys_low,
                             b, b_stat_up, b_stat_low, b_sys_up, b_sys_low,
                             ax, label ):
    a_up_total = np.sqrt(a_stat_up**2 + a_sys_up**2)
    a_low_total = np.sqrt(a_stat_low**2 + a_sys_low**2)
    b_up_total = np.sqrt(b_stat_up**2 + b_sys_up**2)
    b_low_total = np.sqrt(b_stat_low**2 + b_sys_low**2)
    ax.errorbar(a, b, xerr=[[a_low_total], [a_up_total]], yerr=[[b_low_total], [b_up_total]], fmt="o", ms=5, label=label)


def _add_literature_values(ax2, runner):
    """Add literature values to the plot based on the rate function name."""
    if runner.rate_function_name == "power_law":
        _two_param_plot_stat_sys(2.045, 1.845, 1.96, 2.11, 0.86, 2.005e-5, 5.2e-5, 1.6e-5, 0,0, ax2, label = "O14")
        ax2.errorbar(1.82, 2e-5, yerr=.32 * 1e-5, xerr=.386, fmt="o", ms=5, label="L20")
        ax2.errorbar(1.7, 2.27e-5, yerr=0.19e-5, xerr=0.21, color="cyan", fmt="o", ms=5, label="F19")
        ax2.errorbar(2.04, 2.32e-5, xerr=0.9, yerr=0.15e-5, color = "green", fmt="o", ms=5, label="D10")
        _two_param_plot_stat_sys(2.11, 0.28, 0.28, 0, 0, 1.7e-5, 0.3e-5, 0.3e-5, 0, 0, ax2, "P12")

        ax2.set_xlim(1.65, 2.3)
        ax2.set_ylim(1.6e-5, 2.7e-5)
    if "AplusB" in runner.rate_function_name:
        ax2.errorbar(9.3e-4, 2.8e-14, xerr=3.1e-4,
                    yerr=1.2e-14, color="magenta", fmt="o", ms=5, label="D08)")
        ax2.errorbar(3.3e-4, 1.9e-14, xerr=0.2e-4, yerr=0.1e-14, color = "red", fmt="o", ms=5, label = "P12")
        ax2.errorbar(5.4e-4, 1.5e-14, xerr=2e-4, yerr=0.7e-14, color = "cyan", fmt="o", ms=5, label = "K08")
        ax2.errorbar(3.9e-4, 5.3e-14, xerr=0.7e-4, yerr=1.1e-14, color = "green", fmt="o", ms=5, label = "S06")
    if "power_law_dtd" in runner.rate_function_name:
        #ax2.errorbar(2.11e-13, -1.13,  yerr=0.05,xerr=.05e-13, label = "Wiseman (2020)", color = "C0", fmt="o", ms=5)
        results_dict = {"G11": (-1.1, 0.1),
                        "P12": (-0.98, 0.05),
                        "M12": (-1.12, 0.08),
                        "W21": (-1.13, 0.05),}
        for i, (label, (beta, unc)) in enumerate(results_dict.items()):
            ax2.axhline(beta, color="C"+str(i + 3), label = label + " $\sigma$ = " + str(unc))
            xlim = ax2.get_xlim()
            #ax2.fill_between([xlim[0], xlim[1]], beta - unc, beta + unc, color="C"+str(i), alpha=0.2)
    if "prompt_fraction" in runner.rate_function_name:
        plt.axvline(0.59, color = "white", linestyle = "--", label = "Simulated Value")
        plt.axhline(1.38e-4, color = "white", linestyle = "--")



def generate_chi2_map(runner, survey, index, n_samples=50, extent=[1.4, 2.0, 2.0e-5, 2.6e-5], csfr = None):
    """Generate an array of chi2 values over a grid of alpha and beta values for a given survey.
    For now, this only works for the power law fit function.
    Inputs
    ------
    survey : str
        Name of the survey.
    """
    fit_args_dict = runner.fit_args_dict
    chi2_map = np.empty((n_samples, n_samples))
    z_centers = runner.fit_args_dict["z_centers"][survey]

    if len(fit_args_dict["N_gen"][survey]) != len(z_centers):
        num_surveys = len(fit_args_dict["N_gen"][survey]) / len(z_centers)
        if num_surveys % 1 != 0:
            raise ValueError("N_gen length is not a multiple of z_centers length!")
        z_centers = np.tile(z_centers, int(num_surveys))
        logging.debug("updated z_centers for chi2 map:", z_centers)

    param_names = runner.param_names
    if param_names is None:
        param_names = ["param_" + str(i) for i in range(len(runner.x0))]


    rate_function = runner.rate_function if csfr is None else runner.rate_functions[csfr]

    for i, a in enumerate(np.linspace(extent[2], extent[3], n_samples)):
        for j, b in enumerate(np.linspace(extent[0], extent[1], n_samples)):

            if survey == "combined":
                n_data = fit_args_dict["n_data"][survey]
            else:
                n_data = fit_args_dict["n_data"][survey][index]
            values = (a, b)
            if len(param_names) > 2:
                values = (a, b) + tuple(runner.results[survey][0][param_names[2:]].values[0])  # Keep other params at result value.
            chi2_result = chi2(values, fit_args_dict["null_counts"][survey],
                                fit_args_dict["f_norm"][survey],
                                z_centers,
                                fit_args_dict["eff_ij"][survey],
                                n_data,
                                rate_function,
                                fit_args_dict["cov_sys"][survey])
            # Note this is now unsquared
            chi2_map[i][j] = np.sum(chi2_result)
    return chi2_map


def _sanity_plots(survey, runner):
    """Generate sanity check plots for the datasets."""
    LaurenNicePlots()
    plt.clf()
    # for surv in ["DUMP_ALL", "SIM_ALL", "DATA_ALL_1"]:
    #     if surv == "DUMP_ALL":
    #         plt.subplot(1, 2, 1)
    #     if surv == "DATA_ALL_1":
    #         plt.subplot(1, 2, 2)
    #     zcol = runner.datasets[f"{survey}_{surv}"].z_col
    #     try:
    #         plt.scatter(runner.datasets[f"{survey}_{surv}"].df["SIM_PEAKMJD"], runner.datasets[f"{survey}_{surv}"].df[zcol], alpha=0.5, label=surv)
    #     except Exception as _:
    #         plt.scatter(runner.datasets[f"{survey}_{surv}"].df["PEAKMJD"], runner.datasets[f"{survey}_{surv}"].df[zcol], alpha=0.5, label=surv)
    # plt.xlabel("Peak MJD")
    # plt.ylabel("Recovered Redshift")
    # plt.legend()
    # #plt.yscale("log")
    # plt.savefig(f"sanity_check_peakmjd_{survey}.png")


    plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(2, 1, 1)
    plt.tight_layout(pad=3.0)

    bins = np.linspace(np.min(runner.datasets[f"{survey}_DUMP_ALL"].df[runner.datasets[f"{survey}_DUMP_ALL"].z_col]),
                        np.max(runner.datasets[f"{survey}_DUMP_ALL"].df[runner.datasets[f"{survey}_DUMP_ALL"].z_col]), 20)

    labels = ["Uncut Simulation CC", "Uncut Simulation IA", "Simulated Detected IA", "Simulated Detected CC"]
    for i, ds in enumerate([f"{survey}_DUMP_CC", f"{survey}_DUMP_IA", f"{survey}_SIM_IA", f"{survey}_SIM_CC"]):
        data = runner.datasets[ds].df
        zcol = runner.datasets[ds].z_col
        plt.hist(data[zcol], bins=bins, alpha=1.0, label=labels[i], histtype="step", linewidth=2)

    plt.xlabel("Redshift")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend(fontsize=12)
    plt.subplot(2, 1, 2, sharex=ax1)

    bins = np.linspace(np.min(runner.datasets[f"{survey}_DUMP_ALL"].df[runner.datasets[f"{survey}_DUMP_ALL"].z_col]),
                        np.max(runner.datasets[f"{survey}_DUMP_ALL"].df[runner.datasets[f"{survey}_DUMP_ALL"].z_col]), 10)

    labels = ["Uncut Simulation IA+CC", "Simulated Detected IA+CC", f"{survey} Data"]
    for i, ds in enumerate([f"{survey}_DUMP_ALL", f"{survey}_SIM_ALL", f"{survey}_DATA_ALL_1"]):
        data = runner.datasets[ds].df
        zcol = runner.datasets[ds].z_col
        if "DATA" in ds:
            plt.hist(data[zcol], bins=bins, alpha=1, label=labels[i], histtype="step", linewidth=2, color="black")
        else:
            plt.hist(data[zcol], bins=bins, alpha=1, label=labels[i], histtype="step", linewidth=2)


    plt.xlabel("Redshift")
    plt.yscale("log")
    plt.ylabel("Counts")
    plt.legend(fontsize=12)
    plt.tight_layout(pad=3.0)
    plt.suptitle(f"Simulation and Data Counts for {survey}", fontsize=16)
    path = f"debug_plots/sanity_check_counts_{survey}.png"
    logging.debug(f"Saving sanity check plots {path}")
    plt.savefig(path)

    plt.clf()
    bins = np.linspace(0, 1, 20)
    labels = ["Simulated Detected IA+CC", f"{survey} Data"]
    for i, ds in enumerate([f"{survey}_SIM_ALL", f"{survey}_DATA_ALL_1"]):
        data = runner.datasets[ds].df
        zcol = runner.datasets[ds].z_col
        scone_col = runner.datasets[ds].scone_col
        if "DATA" in ds:
            plt.hist(data[scone_col], bins=bins, alpha=1, label=labels[i], histtype="step", linewidth=2, color="black")
        else:
            plt.hist(data[scone_col], bins=bins, alpha=1, label=labels[i], histtype="step", linewidth=2)
    plt.xlabel("Scone Probability")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend()
    path = f"debug_plots/sanity_check_scone_{survey}.png"

    logging.debug(f"Saving sanity check plots {path}")
    plt.savefig(path)

def _transfer_matrix_plot(eff_ij, z_bins, survey):
    LaurenNicePlots()
    plt.clf()
    plt.figure(figsize=(7, 6), dpi = 200)
    plt.imshow(eff_ij[1:-1, :].T, origin="lower", aspect="auto",
            extent=[z_bins[0], z_bins[-1], z_bins[0], z_bins[-1]],
            vmin=0, vmax=np.max(eff_ij)
            )
    plt.colorbar(label=r"Efficiency (n$_{\mathrm{obs}}$ / n$_{\mathrm{sim}}$)")
    plt.title(f"Efficiency Matrix for {survey}")
    plt.ylabel("Recovered Redshift")
    plt.xlabel("True Redshift")
    path = f"plots/efficiency_matrix_{survey}.png"
    logging.debug(f"Saving efficiency matrix plot to {path}")
    plt.savefig(path)