# Standard Library
import yaml

import glob
import logging
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pathlib
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.sparse import block_diag
from scipy import stats


# Astronomy
from astropy.cosmology import LambdaCDM
from astropy.io import fits

# Sauron modules
from funcs import (power_law, turnover_power_law, chi2, calculate_covariance_matrix_term,
                   rescale_CC_for_cov,
                   calculate_null_counts, chi2_to_sigma)
from SN_dataset import SN_dataset

# Get the matplotlib logger
matplotlib_logger = logging.getLogger('matplotlib')

# Set the desired logging level (e.g., INFO, WARNING, ERROR, CRITICAL)
matplotlib_logger.setLevel(logging.WARNING)
#logger = logging.getLogger(__name__)


from matplotlib import rcParams
import matplotlib as mpl
def update_rcParams(key, val):
    if key in rcParams:
        rcParams[key] = val

def LaurenNicePlots():
    update_rcParams('font.size', 10)
    update_rcParams('font.family', 'serif')
    update_rcParams('xtick.major.size', 8)
    update_rcParams('xtick.labelsize', 'large')
    update_rcParams('xtick.direction', "in")
    update_rcParams('xtick.minor.visible', True)
    update_rcParams('xtick.top', True)
    update_rcParams('ytick.major.size', 8)
    update_rcParams('ytick.labelsize', 'large')
    update_rcParams('ytick.direction', "in")
    update_rcParams('ytick.minor.visible', True)
    update_rcParams('ytick.right', True)
    update_rcParams('xtick.minor.size', 4)
    update_rcParams('ytick.minor.size', 4)
    update_rcParams('xtick.major.pad', 10)
    update_rcParams('ytick.major.pad', 10)
    update_rcParams('legend.numpoints', 1)
    update_rcParams('mathtext.fontset', 'cm')
    update_rcParams('mathtext.rm', 'serif')
    update_rcParams('axes.labelsize', 'x-large')
    update_rcParams('lines.marker', 'None')
    update_rcParams('lines.markersize', 1)
    update_rcParams('lines.markeredgewidth', 1.0)
    update_rcParams('lines.markeredgecolor', 'auto')

    #cycle_colors = ['navy', 'maroon','darkorange', 'darkorchid', 'darkturquoise', 'darkmagenta', '6FADFA','7D7D7D','black']
    cycle_colors = ["348ABD", "A60628", "7A68A6", "467821", "D55E00", "CC79A7", "56B4E9", "009E73", "F0E442", "0072B2"]
    # cycle_colors = ['9F6CE6','FF984A','538050','6FADFA','7D7D7D','black']
    cycle_markers = ['o','^','*','s','X','d', '1','2', '3']
    # cycle_colors = ['darkorchid','darkorange','darkturquoise']
    # cycle_markers = ['o','^','*']
    #+ mpl.cycler(marker=cycle_markers)
    update_rcParams('axes.prop_cycle', mpl.cycler(color=cycle_colors) )


# Configure the basic logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cosmo = LambdaCDM(H0=70, Om0=0.315, Ode0=0.685)
# Cosmology parameters updated from Om0=0.3, Ode0=0.7 to Om0=0.315, Ode0=0.685 (Planck-like values)
# This change was made to match SNANA. If you require the previous values for consistency, revert to Om0=0.3, Ode0=0.7.

func_name_dictionary = {
    "power_law": power_law,
    "turnover_power_law": turnover_power_law,
    "dual_power_law": turnover_power_law
}

default_x0_dictionary = {
    "power_law": (2.27e-5, 1.7),
    "turnover_power_law": (1, 0, 1, -2),
    "dual_power_law": (1, 0, 1, -2)
}


class sauron_runner():
    """Class to run SAURON analysis."""
    def __init__(self, args):
        self.args = args
        self.fit_args_dict = {}
        self.fit_args_dict['N_gen'] = {}
        self.fit_args_dict['null_counts'] = {}
        self.fit_args_dict['eff_ij'] = {}
        self.fit_args_dict['f_norm'] = {}
        self.fit_args_dict['n_data'] = {}
        self.fit_args_dict['cov_sys'] = {}
        self.fit_args_dict['cc_are_sep'] = {}
        self.fit_args_dict['z_bins'] = {}
        self.fit_args_dict['z_centers'] = {}
        self.fit_args_dict['n_datasets'] = {}
        self.fit_args_dict['simulated_rate_function'] = {}
        self.fit_args_dict['rate_params'] = {}
        self.results = {}
        self.final_counts = {}

    def parse_global_fit_options(self):
        """ Parse global fit options (I.e. those that apply to all surveys) from the config file."""
        with open(self.args.config, 'r') as f:
            files_input = yaml.safe_load(f)
        fit_options = files_input.get("FIT_OPTIONS", {})
        self.rate_function = func_name_dictionary.get(fit_options.get("RATE_FUNCTION"), None)
        if self.rate_function is None:
            logging.warning("No valid RATE_FUNCTION specified in FIT_OPTIONS. Defaulting to power_law.")
            self.rate_function = power_law
        self.x0 = default_x0_dictionary.get(fit_options.get("RATE_FUNCTION"), (2.27e-5, 1.7))

        if "X0" in fit_options:
            self.x0 = fit_options["X0"]
        else:
            logging.warning(f"No X0 specified in FIT_OPTIONS. Using default initial guess {self.x0}.")

    def parse_survey_fit_options(self, args_dict, survey):
        """ Parse survey-specific fit options from the config file.

        Inputs
        ------
        args_dict : dict
            Dictionary of fit options for the survey, loaded from the config file.

        survey : str
            Name of the survey.
        """
        self.fit_args_dict["cc_are_sep"][survey] = args_dict.get("CC_ARE_SEPARATE", None)
        logging.debug(f"Setting CC_ARE_SEPARATE for {survey} to {args_dict.get('CC_ARE_SEPARATE', None)}")
        self.fit_args_dict["z_bins"][survey] = np.arange(0.0, 1.4, 0.1)
        if "Z_BINS" in args_dict:
            if isinstance(args_dict["Z_BINS"], list):
                self.fit_args_dict["z_bins"][survey] = np.array(args_dict["Z_BINS"])
            elif isinstance(args_dict["Z_BINS"], int):
                if "MIN_Z" not in args_dict or "MAX_Z" not in args_dict:
                    logging.warning("When specifying Z_BINS as an integer, MIN_Z and MAX_Z must also be specified."
                                    " Defaulting to MIN_Z=0 and MAX_Z=1.4")
                min_z = args_dict.get("MIN_Z", 0)
                max_z = args_dict.get("MAX_Z", 1.4)
                self.fit_args_dict["z_bins"][survey] = np.linspace(min_z, max_z, args_dict["Z_BINS"] + 1)

        else:
            logging.warning(f"No Z_BINS specified in FIT_OPTIONS. Using default z_bins for {survey}:")

        self.fit_args_dict["z_centers"][survey] = (self.fit_args_dict["z_bins"][survey][1:] +
                                                   self.fit_args_dict["z_bins"][survey][:-1]) / 2

        simulated_rate_func = args_dict.get("RATE_FUNC", None)
        if simulated_rate_func is not None:
            self.fit_args_dict["simulated_rate_function"][survey] = simulated_rate_func
            self.fit_args_dict["simulated_rate_function"]["combined"] = simulated_rate_func
            logging.warning("Using specified RATE_FUNC for the surveys to define combined rate func."
                            " This needs to be fixed later.")
        else:
            raise ValueError(f"RATE_FUNC must be specified in FIT_OPTIONS for {survey}.")

        simulated_rate_params = args_dict.get("RATE_PARAMS", None)
        if simulated_rate_params is not None:
            simulated_rate_params = [float(i) for i in simulated_rate_params.split(",")]
            self.fit_args_dict["rate_params"][survey] = simulated_rate_params
            self.fit_args_dict["rate_params"]["combined"] = simulated_rate_params
            logging.warning("Using specified RATE_PARAMS for the surveys to define combined rate params."
                            " This needs to be fixed later.")
        else:
            raise ValueError(f"RATE_PARAMS must be specified in FIT_OPTIONS for {survey}.")

    def unpack_dataframes(self):
        """Load dataframes pointed to in the config file into SN_dataset objects. This includes DUMP, SIM,
        and DATA files, splitting and combining them based on IA vs CC as necessary.
        TODO: Turn this into smaller functions for readability.
        """

        files_input = yaml.safe_load(open(self.args.config, 'r'))
        surveys = list(files_input.keys())
        if "FIT_OPTIONS" in surveys:
            surveys.remove("FIT_OPTIONS")
        datasets = {}
        # Unpack dataframes into SN_dataset objects
        for survey in surveys:
            survey_dict = files_input[survey]
            fit_args_dict = survey_dict.get("FIT_OPTIONS", {})
            self.parse_survey_fit_options(fit_args_dict, survey)
            for i, file in enumerate(list(survey_dict.keys())):
                if "DUMP" not in file and "SIM" not in file and "DATA" not in file:
                    continue  # Skip non-data files

                sntype = "IA" if "IA" in file else "CC"

                if isinstance(survey_dict[file], dict):
                    zcol = survey_dict[file].get("ZCOL", None)
                else:
                    zcol = None

                # Either use the paths provided or glob the directory provided
                if survey_dict[file].get('PATH') is not None:
                    paths = survey_dict[file]['PATH']
                    paths = [paths] if not isinstance(paths, list) else paths  # Make it a list for later
                    paths = glob.glob(paths[0]) if len(paths) == 1 else paths  # Check to see if they meant to glob
                    if len(paths) == 0:
                        raise FileNotFoundError(f"No files found for {survey} {file} with path {survey_dict[file]['PATH']}")
                elif survey_dict[file].get('DIR') is not None:
                    paths = []
                    for dir in survey_dict[file]['DIR']:
                        paths.extend(glob.glob(dir + "/**/*.gz"))
                        paths.extend(glob.glob(dir + "*.gz"))  # This extension can't be hardcoded
                    logging.info(f"Found {len(paths)} files in {survey_dict[file]['DIR']}")

                logging.debug(f"Paths for {survey} {file}: {paths}")

                if "DATA" in file:
                    for i, path in enumerate(paths):
                        cuts = survey_dict.get("CUTS", None)
                        sntypecol = survey_dict[file].get("SNTYPECOL", None)
                        datasets[survey+"_"+file+"_"+str(i+1)] = SN_dataset(path,
                                                                            sntype, data_name=survey+"_"+file,
                                                                            zcol=zcol, cuts=cuts,
                                                                            sntypecol=sntypecol)
                    n_datasets = len(paths)
                    self.fit_args_dict["n_datasets"][survey] = n_datasets
                    self.fit_args_dict["n_datasets"]["combined"] = 1  # This needs to be fixed later TODO
                    logging.info(f"Found {n_datasets} data sets for {survey}")

                else:
                    cuts = survey_dict.get("CUTS", None)
                    true_z_col = survey_dict[file].get("TRUEZCOL", None)
                    sntypecol = survey_dict[file].get("SNTYPECOL", None)
                    datasets[survey+"_"+file] = SN_dataset(paths, sntype, data_name=survey+"_"+file, zcol=zcol,
                                                           cuts=cuts, true_z_col=true_z_col, sntypecol=sntypecol)
                    #logging.debug(f"z bin counts for {survey}_{file}: "
                    #              f"{datasets[survey+'_'+file].z_counts(self.fit_args_dict['z_bins'][survey])}")
                    #logging.debug(f"True z col for {survey}_{file}: {datasets[survey+'_'+file].true_z_col}")
                    logging.debug(f"scone col for {survey}_{file}: {getattr(datasets[survey+'_'+file], 'scone_col', None)}")


            if self.fit_args_dict["cc_are_sep"].get(survey) is None:
                self.fit_args_dict["cc_are_sep"][survey] = True
                logging.warning("CC_ARE_SEPARATE not specified in config file for "f"{survey}. Defaulting to True.")
            # Combine IA and CC files if they are separate


            if self.fit_args_dict["cc_are_sep"][survey]:

                if not self.args.cheat_cc and datasets.get(f"{survey}_DUMP_CC") is not None:
                    logging.info("Combining IA and CC dump and sim files..")
                    datasets[f"{survey}_DUMP_ALL"] = datasets[f"{survey}_DUMP_IA"].combine_with(
                        datasets[f"{survey}_DUMP_CC"], "all", data_name=survey+"_DUMP_ALL")
                    datasets[f"{survey}_SIM_ALL"] = datasets[f"{survey}_SIM_IA"].combine_with(
                        datasets[f"{survey}_SIM_CC"], "all", data_name=survey+"_SIM_ALL")
                    # Data files may be combined even if sim and dump are not
                    if datasets.get(f"{survey}_DATA_CC_1") is not None:
                        for i in range(n_datasets):
                            datasets[f"{survey}_DATA_ALL_{i+1}"] = datasets[f"{survey}_DATA_IA_"+str(i+1)].combine_with(
                                datasets[f"{survey}_DATA_CC_"+str(i+1)], "all", data_name=survey+f"_DATA_ALL_{i+1}")
                else:
                    if self.args.cheat_cc:
                        logging.info("Skipping combining IA and CC files because --cheat_cc was set.")
                    elif datasets.get(f"{survey}_DUMP_CC") is None:
                        logging.warning(f"Couldn't find {survey}_DUMP_CC to combine with IA file.")

            # Otherwise, if they aren't seperate, we need to split DUMP and SIM into IA and CC
            else:
                # Validate that combined DUMP and SIM datasets exist when corecollapse_are_separate is False
                try:
                    dump_all_dataset = datasets[f"{survey}_DUMP_ALL"]
                    sim_all_dataset = datasets[f"{survey}_SIM_ALL"]
                except KeyError as exc:
                    raise KeyError(
                        f"Couldn't find {survey}_DUMP_ALL or {survey}_SIM_ALL in datasets. "
                        "If your DUMP and SIM files are already separate for IA and CC, "
                        "set corecollapse_are_separate to True in the config."
                    ) from exc
                # Validate that required config keys are present to split combined samples into IA and CC
                try:
                    dump_sn_col = survey_dict["DUMP_ALL"]["SNTYPECOL"]
                    dump_ia_vals = survey_dict["DUMP_ALL"]["IA_VALS"]
                    sim_sn_col = survey_dict["SIM_ALL"]["SNTYPECOL"]
                    sim_ia_vals = survey_dict["SIM_ALL"]["IA_VALS"]
                except KeyError as exc:
                    raise KeyError(
                        f"Missing configuration for survey '{survey}' when corecollapse_are_separate is False. "
                        "Expected keys in the survey's config section: "
                        "'DUMP_ALL.SNTYPECOL', 'DUMP_ALL.IA_VALS', "
                        "'SIM_ALL.SNTYPECOL', and 'SIM_ALL.IA_VALS'. "
                        "These are needed to separate DUMP and SIM into IA and CC."
                    ) from exc
                datasets[f"{survey}_DUMP_IA"], datasets[f"{survey}_DUMP_CC"] = dump_all_dataset.split_into_IA_and_CC(
                    dump_sn_col, dump_ia_vals)
                datasets[f"{survey}_SIM_IA"], datasets[f"{survey}_SIM_CC"] = sim_all_dataset.split_into_IA_and_CC(
                    sim_sn_col, sim_ia_vals)

            logging.debug(f"Datasets keys after unpacking: {list(datasets.keys())}")
            if self.args.cheat_cc and datasets.get(f"{survey}_DATA_IA_1") is None:
                data_sn_col = survey_dict["DATA_ALL"]["SNTYPECOL"]
                ia_vals_data = survey_dict["DATA_ALL"]["IA_VALS"]
                for i in range(n_datasets):
                    logging.debug("Splitting DATA into IA and CC using cheat mode...")
                    datasets[f"{survey}_DATA_IA_"+str(i+1)], _ =\
                        datasets[f"{survey}_DATA_ALL_"+str(i+1)].split_into_IA_and_CC(
                            data_sn_col, ia_vals_data)


                    # data_df = datasets[f"{survey}_DATA_ALL_"+str(i+1)].df
                    # data_ia_df = data_df[data_df[data_sn_col].isin(ia_vals_data)]
                    # datasets[f"{survey}_DATA_IA_"+str(i+1)] =\
                    #     SN_dataset(data_ia_df, "IA", zcol=datasets[f"{survey}_DATA_ALL_"+str(i+1)].z_col,
                    #                data_name=survey+f"_DATA_IA_{i+1}")
                    logging.debug("z bin counts for {survey}_DATA_IA_{i+1}: "
                                  f"{datasets[f'{survey}_DATA_IA_'+str(i+1)].z_counts(
                                    self.fit_args_dict['z_bins'][survey])}")

        self.datasets = datasets
        self.surveys = surveys

        # for d in datasets:
        #     counts = datasets[d].z_counts(self.fit_args_dict['z_bins'][survey])
        #     assert np.size(np.where(counts == 0)[0]) <= 2, f"{d} has several zero count bins! {counts}"

        return datasets, surveys

    def get_counts(self, survey):
        """Get counts of generated and initialize bins for a given survey.
        Inputs
        ------
        survey : str
            Name of the survey.
        """
        z_bins = self.fit_args_dict['z_bins'][survey]
        self.fit_args_dict['N_gen'][survey] = self.datasets[f"{survey}_DUMP_IA"].z_counts(z_bins)
        self.results[survey] = []

        self.final_counts[survey] = {}
        self.final_counts["combined"] = {}


    def calculate_transfer_matrix(self, survey):
        """Calculate the transfer matrix, epsilon_ij, for a given survey.
        Inputs
        ------
        survey : str
            Name of the survey."""
        dump = self.datasets[f"{survey}_DUMP_IA"]
        sim = self.datasets[f"{survey}_SIM_IA"]
        z_bins = self.fit_args_dict['z_bins'][survey]

        logging.info(f"Using true col: {sim.true_z_col} and recovered col: {sim.z_col}")
        simulated_events = sim.df
        sim_z_col = sim.z_col
        true_z_col = sim.true_z_col

        z_bins = self.fit_args_dict['z_bins'][survey]
        dump_counts = dump.z_counts(z_bins)

        logging.debug(f"Dump counts: {dump_counts}")

        logging.debug(f"true z col counts: {np.histogram(simulated_events[true_z_col], bins=z_bins)[0]}")

        z_bins_expanded = np.concatenate(([-np.inf], z_bins, [np.inf]))

        num, _, _ = np.histogram2d(simulated_events[true_z_col], simulated_events[sim_z_col],
                                   bins=[z_bins_expanded, z_bins])

        if np.any(dump_counts == 0):
            logging.warning("Some redshift bins have zero simulated events! This may cause issues.")
            bad_bins = np.where(dump_counts == 0)[0]
            upper_bad_bins = bad_bins + 1
            unique_bad_bins = np.sort(np.unique(np.concatenate((bad_bins, upper_bad_bins))))
            logging.warning("Specifically, these are the bin edges of the zero count bins:", z_bins[unique_bad_bins])

        eff_ij = num/dump_counts

        self.fit_args_dict['eff_ij'][survey] = eff_ij

        if self.args.debug:
            plt.clf()
            plt.imshow(eff_ij, origin='lower', aspect='auto',
                       extent=[z_bins[0], z_bins[-1], z_bins[0], z_bins[-1]],
                       vmin=0, vmax=1)
            plt.colorbar(label="Efficiency")
            plt.title(f"Transfer Matrix for {survey}")
            plt.xlabel("Reconstructed Redshift")
            plt.ylabel("True Redshift")
            plt.savefig(f"transfer_matrix_{survey}.png")

        return eff_ij

    def fit_rate(self, survey):
        """Actually fit the rate parameters for a given survey.
        Inputs
        ------
        survey : str or list
            Name of the survey, or list of surveys to fit together.
        """
        # How will this work when I am fitting a non-power law?
        # How do I get the inherent rate in the simulation? Get away from tracking simulated efficiency.
        if not isinstance(survey, list):
            survey = [survey]

        logging.info(f"Fitting survey(s): {survey} ###########################")
        z_centers = []
        z_bins_list = []
        f_norms = []
        for s in survey:
            z_bins = self.fit_args_dict['z_bins'][s]
            logging.debug(f"z_bins for survey {s}: {z_bins}")
            z_bins_list.extend(z_bins)
            z_centers.extend(z_bins[:-1]/2 + z_bins[1:]/2)
            f_norm = self.fit_args_dict['f_norm'][s]
            f_norms.extend(np.repeat(f_norm, len(z_bins)-1))

        z_centers = np.array(z_centers)
        f_norms = np.array(f_norms)
        # This needs to be done survey by survey because f_norm is per survey
        n_data = np.concatenate([self.fit_args_dict['n_data'][s] for s in survey])
        N_gen = np.concatenate([self.fit_args_dict['N_gen'][s] for s in survey])
        eff_ij_list = [self.fit_args_dict['eff_ij'][s] for s in survey]
        eff_ij = block_diag(eff_ij_list).toarray()
        cov_sys_list = [self.fit_args_dict['cov_sys'][s] for s in survey]
        for i, c in enumerate(cov_sys_list):
            s = survey[i]
            size = len(self.fit_args_dict['z_centers'][s])
            if c is None:
                cov_sys_list[i] = np.zeros((size, size))
        cov_sys = block_diag(cov_sys_list).toarray()

        survey = survey[0] if len(survey) == 1 else "combined"
        # Note this only allows for individual surveys or all, no subsets. Fix this later.
        if survey == "combined":
            self.fit_args_dict['f_norm'][survey] = f_norms
            self.fit_args_dict['z_bins'][survey] = z_bins_list
            self.fit_args_dict['n_data'][survey] = n_data
            self.fit_args_dict['N_gen'][survey] = N_gen
            self.fit_args_dict['eff_ij'][survey] = eff_ij
            self.fit_args_dict['cov_sys'][survey] = cov_sys
            self.fit_args_dict['z_centers'][survey] = z_centers
            self.fit_args_dict['n_datasets'][survey] = 1
            self.final_counts[survey] = {}
            self.results[survey] = []

        # The above are only really needed for debugging.

        logging.warning("This doesn't work for multiple surveys yet!")

        true_rate_function = func_name_dictionary.get(self.fit_args_dict["simulated_rate_function"][survey])
        logging.debug(f"Using true rate function: {true_rate_function}")
        logging.debug(f"With params: {self.fit_args_dict['rate_params'][survey]}")
        null_counts = calculate_null_counts(z_bins_list, z_centers, N_gen, cosmo=cosmo,
                                            true_rate_function=true_rate_function,
                                            rate_params=self.fit_args_dict["rate_params"][survey])
        self.fit_args_dict['null_counts'][survey] = null_counts

        logging.debug(f"data counts: {n_data}")

        logging.debug(f"x0: {self.x0}")

        fJ_0 = self.x0[0] * (1 + z_centers)**self.x0[1]
        x0_counts = np.sum(null_counts * eff_ij * f_norms * fJ_0, axis=0)
        logging.debug(f"Initial predicted counts (x0): {x0_counts}")

        logging.debug(f"Total counts in dataset {survey}: {np.sum(n_data)}")

        fit_method = "minimize"
        N = len(z_centers)  # number of data points
        n = len(self.x0)  # number of parameters

        if fit_method == "minimize":
            result = minimize(
                        chi2,
                        x0=self.x0,
                        args=(null_counts, f_norms, z_centers, eff_ij,
                              n_data, self.rate_function, cov_sys),
                        method=None
                    )
            fit_params = result.x

            logging.debug(f"Least Squares Result: {fit_params}")

            # This calculation of residual variance is only valid if minimizing sum of squares

            def errFit(hess_inv, resVariance):
                return np.sqrt(np.diag(hess_inv * resVariance))

            residual_variance = result.fun / (N - n)
            dFit = errFit(result.hess_inv, residual_variance)
            logging.debug(f"Standard errors Stack Overflow: {dFit}")
            cov_x = result.hess_inv * residual_variance
            chi_squared = result.fun

        elif fit_method == "curve_fit":
            from scipy.optimize import curve_fit

            fJ_0 = self.rate_function(z_centers, self.x0)
            Ei = np.sum(null_counts * eff_ij * f_norm * fJ_0, axis=0)
            var_Ei = np.abs(Ei)
            var_Si = np.sum(null_counts * eff_ij * f_norm**2 * fJ_0**2, axis=0)

            cov_stat = np.diag(var_Ei + var_Si)

            cov = cov_stat + cov_sys
            logger.debug(f"Sys Covariance Matrix Diag: {np.diag(cov_sys)}")
            logger.debug(f"Stat Covariance Matrix Diag: {np.diag(cov_stat)}")
            logger.debug(f"Covariance Matrix Diag: {np.diag(cov)}")

            popt, pcov = curve_fit(
                lambda z, a, b: np.sum(null_counts * eff_ij * f_norms *
                                      self.rate_function(z, [a, b]), axis=0), sigma=cov,
                                      absolute_sigma=False,
                xdata=z_centers,
                ydata=n_data,
                p0=self.x0
            )

            fit_params = popt
            cov_x = pcov
            chi_squared = chi2(fit_params, null_counts, f_norms, z_centers,
                               eff_ij, n_data, self.rate_function, cov_sys)
            logging.debug(f"Curve Fit Result: {fit_params}")
            logging.debug(f"Covariance Matrix from curve_fit: {cov_x}")
            logging.debug(f"Standard errors from curve_fit: {np.sqrt(np.diag(cov_x))}")

        fJ = self.rate_function(z_centers, fit_params)
        Ei = np.sum(null_counts * eff_ij * f_norms * fJ, axis=0)

        logging.debug(f"Predicted Counts Ei: {Ei}")

        # Estimate errors on Ei
        alpha_draws = np.random.normal(fit_params[0], np.sqrt(cov_x[0, 0]), size=1000)
        beta_draws = np.random.normal(fit_params[1], np.sqrt(cov_x[1, 1]), size=1000)
        fJ_draws = alpha_draws * (1 + z_centers)[:, np.newaxis]**beta_draws
        N_gen = N_gen[:, np.newaxis]  # for broadcasting
        eff_ij = np.repeat(eff_ij[:, :, np.newaxis], 1000, axis=2)
        f_norms = np.atleast_1d(f_norms)
        f_norms = f_norms[:, np.newaxis]  # for broadcasting

        # I am not confident this is correct. Check later.

        Ei_draws = np.sum(N_gen * eff_ij * f_norms * fJ_draws, axis=0)
        Ei_err = np.std(Ei_draws, axis=1)

        self.final_counts[survey]["predicted_counts"] = Ei
        self.final_counts[survey]["x0_counts"] = x0_counts
        self.final_counts[survey]["observed_counts"] = n_data
        self.final_counts[survey]["predicted_counts_err"] = Ei_err
        self.n_data = n_data

        self.final_counts[survey]["result"] = fit_params
        self.final_counts[survey]["covariance"] = cov_x
        self.final_counts[survey]["chi"] = chi_squared
        # Should there not be an index here?

        return fit_params, chi_squared, Ei, cov_x, Ei_err

    def save_results(self):
        """Save results to output file specified in args."""
        output_df = pd.DataFrame()
        surveys = self.results.keys()
        for survey in surveys:
            results = self.results[survey]
            results = [results] if not isinstance(results, list) else results
            for i, result in enumerate(results):
                output_df = pd.concat([output_df, result], ignore_index=True)

        output_path = self.args.output
        logging.info(f"Saving to {output_path}")
        output_df.to_csv(output_path, index=False)

    def calculate_CC_contamination(self, PROB_THRESH, index, survey, datasets=None, debug=False, method="Lasker"):
        """Calculate CC contamination for a given survey and dataset index.

        Inputs
        ------
        PROB_THRESH : float
           Probability threshold for classification, this should be the probability the SN is believed to be IA.
        index : int
            Dataset index.
        survey : str
            Name of the survey.
        datasets: dict
            Dictionary of datasets. Note, this is stored in this class as self.datasets, but since this function is also
            used for the rescaling contaminant rates systematic, it is passed as an argument here.
        """
        datasets = self.datasets if datasets is None else datasets
        z_bins = self.fit_args_dict['z_bins'][survey]
        cheat = self.args.cheat_cc
        method = "scone_cut"
        if not cheat and datasets.get(f"{survey}_DUMP_CC") is not None:
            if method == "Lasker":
                IA_frac = (datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH) /
                           datasets[f"{survey}_SIM_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH))
                logging.debug(f"Simulated IA counts {datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH)}")
                logging.debug(f"Simulated ALL counts {datasets[f"{survey}_SIM_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH)}")


                N_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins))
                logging.debug("Total N_data before CC contamination: "
                              f"{datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)}")
                n_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH))

                N_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)
                n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)

                dataset = datasets[f"{survey}_DATA_ALL_{index}"].df



                #import pdb; pdb.set_trace()

                R = n_data / N_data

                #N_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
                #n_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH))

                N_IA_sim = datasets[f"{survey}_SIM_IA"].z_counts(z_bins)
                n_IA_sim = datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH)



                #N_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins))
                #n_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins, prob_thresh=PROB_THRESH))

                N_CC_sim = datasets[f"{survey}_SIM_CC"].z_counts(z_bins)
                n_CC_sim = datasets[f"{survey}_SIM_CC"].z_counts(z_bins, prob_thresh=PROB_THRESH)

                # These lines below are debug and should be removed
                true_IAs_data = dataset[dataset['TYPE'].isin([101, 111])]
                true_CCs_data = dataset[~dataset['TYPE'].isin([101, 111])] # I need to confirm these are the right types
                #for c in true_IAs_data.columns:
                #    logging.debug(c)
                true_IAs_data = true_IAs_data[true_IAs_data["PROB_SCONE"] >= PROB_THRESH]
                true_CCs_data = true_CCs_data[true_CCs_data["PROB_SCONE"] >= PROB_THRESH]

                true_IAs_data = np.histogram(true_IAs_data["zHD"], bins=z_bins, weights=None)[0]
                true_CCs_data = np.histogram(true_CCs_data["zHD"], bins=z_bins, weights=None)[0]
                logging.debug(f"True IA counts in data: {true_IAs_data}")
                logging.debug(f"True CC counts in data: {true_CCs_data}")
                logging.debug(f"True IA fraction in data: {true_IAs_data / (true_IAs_data + true_CCs_data)}")
                logging.debug(f"True scaling: {true_CCs_data / (0.02 * np.sum(N_CC_sim))}")

                logging.debug(f"Calculated R: {R}")
                logging.debug(f"N_IA_sim: {N_IA_sim}, n_IA_sim: {n_IA_sim}")
                logging.debug(f"N_CC_sim: {N_CC_sim}, n_CC_sim: {n_CC_sim}")
                S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)
                logging.debug(f"Calculated contamination scaling S: {S}")
                logging.debug(f"CC_frac before rescaling: {1 - IA_frac}")
                CC_frac = (1 - IA_frac) * S
                logging.debug(f"Calculated CC fraction after rescaling: {CC_frac}")
                IA_frac = np.nan_to_num(1 - CC_frac)
                logging.debug(f"Calculated IA fraction after contamination: {IA_frac}")
                logging.debug(f"True IA fraction in data: {true_IAs_data / (true_IAs_data + true_CCs_data)}")

                inverse_Ia_reduction_Fraction = N_IA_sim / n_IA_sim

                n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac *\
                    inverse_Ia_reduction_Fraction

                if debug:
                    plt.clf()
                    plt.subplot(1, 2, 1)
                    plt.plot(CC_frac, label="CC fraction vs z after contamination")
                    plt.plot(IA_frac, label="IA fraction vs z after contamination")
                    plt.axhline(0, color='k', linestyle='--', lw=1)
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(n_data, label="DATA ALL counts after CC contamination")
                    #n_data_scone_cut = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
                    #plt.plot(n_data_scone_cut, label="DATA ALL counts using scone cut")
                    n_all = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)
                    plt.plot(n_all, label="DATA counts before CC contamination")
                    plt.axhline(0, color='k', linestyle='--', lw=1)
                    logging.debug(f"Calculated n_data after CC contamination: {n_data}")
                    plt.legend()
                    plt.savefig(f"scone_decontamination_{survey}_dataset{index}.png")
            elif method == "scone_cut":
                logging.debug("Performing just a scone cut for decontamination.")
                n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
                bias_correction = datasets[f"{survey}_SIM_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH) / \
                                    datasets[f"{survey}_SIM_IA"].z_counts(z_bins)
                bias_correction = np.nan_to_num(bias_correction, nan=1.0, posinf=1.0, neginf=1.0)
                logger.debug(f"Bias correction factor for scone cut: {1 / bias_correction}")
                n_data /= bias_correction
                logger.debug(f"Calculated n_data after CC contamination using scone cut: {n_data}")

        else:
            if cheat:
                logging.warning("SKIPPING CC CONTAMINATION STEP. USING DATA_IA AS DATA_ALL.")
            else:
                logging.warning(f"Could not find {survey}_DUMP_CC to calculate CC contamination."
                                " Skipping CC contamination step.")

            logging.warning("SKIPPING CC CONTAMINATION STEP. USING DATA_IA AS DATA_ALL.")
            if datasets.get(f"{survey}_DATA_IA_{index}") is not None:
                datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
            else:
                raise notImplementedError("DATA_IA file not found, and cheat_cc is set. Cannot proceed.")
                # If no DATA_IA file exists, filter DATA_ALL for IA SNe only
                # data_sn_col = survey_dict["DATA_ALL"]["SNTYPECOL"]
                # ia_vals_data = survey_dict["DATA_ALL"]["IA_VALS"]
                # datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_ALL"][np.where(
                #     datasets[f"{survey}_DATA_ALL"].df[data_sn_col].isin(ia_vals_data))]

            n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)

        return n_data

    def generate_chi2_map(self, survey, n_samples=50, extent=[1.4, 2.0, 2.0e-5, 2.6e-5]):
        """Generate an array of chi2 values over a grid of alpha and beta values for a given survey.
        For now, this only works for the power law fit function.
        Inputs
        ------
        survey : str
            Name of the survey.
        """

        fit_args_dict = self.fit_args_dict
        chi2_map = np.empty((n_samples, n_samples))
        z_centers = self.fit_args_dict['z_centers'][survey]

        if len(fit_args_dict['N_gen'][survey]) != len(z_centers):
            num_surveys = len(fit_args_dict['N_gen'][survey]) / len(z_centers)
            assert (num_surveys % 1 == 0), "N_gen length is not a multiple of z_centers length!"
            z_centers = np.tile(z_centers, int(num_surveys))
            logging.debug("updated z_centers for chi2 map:", z_centers)

        logging.debug("SANITY CHECK chi2 map values:")
        chi2_result = chi2((2.27e-5, 1.7), fit_args_dict['null_counts'][survey], fit_args_dict['f_norm'][survey],
                           z_centers,
                           fit_args_dict['eff_ij'][survey],
                           fit_args_dict['n_data'][survey],
                           self.rate_function,
                           fit_args_dict['cov_sys'][survey])
        logging.debug(f"chi2 at (2.27e-5, 1.7): {np.sum(chi2_result)}")

        for i, a in enumerate(np.linspace(extent[2], extent[3], n_samples)):
            for j, b in enumerate(np.linspace(extent[0], extent[1], n_samples)):

                values = (a, b)

                chi2_result = chi2(values, fit_args_dict['null_counts'][survey], fit_args_dict['f_norm'][survey],
                                   z_centers,
                                   fit_args_dict['eff_ij'][survey],
                                   fit_args_dict['n_data'][survey],
                                   self.rate_function,
                                   fit_args_dict['cov_sys'][survey])
                # Note this is now unsquared
                chi2_map[i][j] = np.sum(chi2_result)
        return chi2_map

    def summary_plot(self):
        """ Generate a plot of the results, including predicted vs observed counts and chi2 contours."""
        surveys = self.results.keys()
        num_plots = len(surveys) + 1
        sides = int(np.ceil(num_plots/2))
        fig, ax = plt.subplots(2, sides, figsize=(12, 6))
        ax = ax.flatten()
        surveys = list(surveys)
        #if "combined" in surveys:
        #    surveys.remove("combined")
        for i, survey in enumerate(surveys):
            s = survey
            if survey != "combined":
                ax1 = ax[0]
                z_centers = np.array(self.fit_args_dict['z_centers'][survey])

                ax1.errorbar(z_centers, self.final_counts[survey]["predicted_counts"],
                             yerr=self.final_counts[survey]["predicted_counts_err"],  fmt='o',
                             label=f" {survey} Sauron Prediction ")
                ax1.errorbar(z_centers, self.final_counts[survey]["observed_counts"],
                             yerr=np.sqrt(self.final_counts[survey]["observed_counts"]),
                             fmt='o', label=f" {survey} Data")
                # ax1.errorbar(z_centers, self.final_counts[survey]["x0_counts"],
                #              yerr=np.sqrt(self.final_counts[survey]["x0_counts"]),
                #              fmt='o', label=f" {survey} Initial Prediction ")
                ax1.legend()
                ax1.set_xlabel("Redshift")
                ax1.set_ylabel("Counts")
                ax1.set_yscale("log")

            if isinstance(self.results[s], list):
                df = self.results[s][0]
            else:
                df = self.results[s]

            ax2 = ax[i+1]
            extent_chi = [df["beta"][0] - 3 * df["beta_error"][0], df["beta"][0] + 3 * df["beta_error"][0],
                          df["alpha"][0] - 3 * df["alpha_error"][0], df["alpha"][0] + 3 * df["alpha_error"][0]]
            logger.debug(extent_chi)
            chi2_map = self.generate_chi2_map(s, extent=extent_chi)
            # normalized_map = chi2_map # - np.min(chi2_map)   # +1 to avoid log(0)
            chi2_map -= np.min(chi2_map)
            logging.debug(f"min chi2 for {survey}: {np.min(chi2_map)}")

            #sigma_map = chi2_to_sigma(chi2_map, dof=len(z_centers) - 2)
            sigma_map = chi2_map

            im = ax2.imshow(sigma_map, extent=extent_chi, origin='lower', aspect='auto', cmap="plasma")
            #ax2.contour(sigma_map, levels=[1, 2, 3], extent=[1.4, 2, 2.0e-5, 2.6e-5], colors='k', linewidths=1)
            # Δχ² contour levels for 2 parameters (≈1σ, 2σ, 3σ confidence regions; see Numerical Recipes / χ² tables)
            ax2.contour(sigma_map, levels=[2.30, 6.18, 11.83], extent=extent_chi, colors='k', linewidths=1)
            plt.colorbar(im, ax=ax2, label="Delta Chi Squared")
            #ax2.axhline(2.27e-5, color='black', linestyle='--')
            #ax2.axvline(1.7, color='black', linestyle='--', label="Fromhaier")
            ax2.errorbar(df["beta"], df["alpha"], xerr=df["beta_error"], yerr=df["alpha_error"], fmt='o',
                         color='white', ms=10, label=f"Fit results {survey}")
            ax2.errorbar(1.82, 2e-5, yerr=.32 * 1e-5, xerr=.386, color = "red", fmt='o', ms=10, label="Lasker")
            ax2.errorbar(1.7, 2.27e-5, yerr=0.19e-5, xerr=0.21, color='cyan', fmt='o', ms=10, label="Fromhaier")
            ax2.set_xlabel("beta")
            ax2.set_ylabel("alpha")
            ax2.set_xlim(extent_chi[0], extent_chi[1])
            ax2.set_ylim(extent_chi[2], extent_chi[3])
            ax2.legend()

        fig.savefig("summary_plot.png")

    def calculate_covariance(self, PROB_THRESH=0.5):
        """Calculate systematic covariance matrix for each survey.
        Inputs
        ------
        PROB_THRESH : float
            Probability threshold for classification, this should be the probability the SN is believed to be IA.
            This is used for varying the prob thresh in the covariance calculation.
        """
        for survey in self.surveys:
            do_sys_cov = getattr(self.args, "sys_cov", None)
            do_sys_cov = False if do_sys_cov is None else do_sys_cov
            if do_sys_cov:
                cov_thresh = calculate_covariance_matrix_term(self.calculate_CC_contamination, [0.45, 0.5, 0.55],
                                                              self.fit_args_dict["z_bins"][survey], 1, survey)

                xx = [0.05, 0.16, 0.5, 0.84, 0.95] # 5 values between +/- 2 sigma \
                X = stats.norm(loc=1, scale=0.2)
                vals = X.ppf(xx)


                #grid = np.meshgrid(vals, vals, vals, indexing='ij')

                # Create a copy for each unique type of CC contaminant
                val_list = [vals] * len(self.datasets[f"{survey}_SIM_CC"].df.TYPE.unique())

                grid = np.meshgrid(*val_list, indexing='ij')
                # grid0 = grid[0].flatten()
                # grid1 = grid[1].flatten()
                # grid2 = grid[2].flatten()
                seeds = np.array(np.arange(len(grid[0].flatten())))
                # rescale_vals = np.array([grid0, grid1, grid2, seeds]).T

                rescale_vals = [grid[i].flatten() for i in range(len(grid))]
                rescale_vals.append(seeds)
                rescale_vals = np.array(rescale_vals).T

                cov_rate_norm = calculate_covariance_matrix_term(rescale_CC_for_cov, rescale_vals,
                                                                 self.fit_args_dict["z_bins"][survey], PROB_THRESH,
                                                                 1, survey, self.datasets,
                                                                 self.fit_args_dict["z_bins"][survey], False)
                # Hard coding index to one needs to change. TODO: Refactor to avoid hardcoded index value
                #  (currently set to 1). This function should not need index at all.
                cov_sys = cov_thresh + cov_rate_norm

                reduced_cov = np.empty(cov_sys.shape)
                for i in range(cov_sys.shape[0]):
                    for j in range(cov_sys.shape[1]):
                        reduced_cov[i, j] = cov_sys[i, j] / (cov_sys[i, i]**0.5 * cov_sys[j, j]**0.5)
                reduced_cov_thresh = np.empty(cov_thresh.shape)
                for i in range(cov_thresh.shape[0]):
                    for j in range(cov_thresh.shape[1]):
                        reduced_cov_thresh[i, j] = cov_thresh[i, j] / (cov_thresh[i, i]**0.5 * cov_thresh[j, j]**0.5)
                reduced_cov_rate_norm = np.empty(cov_rate_norm.shape)
                for i in range(cov_rate_norm.shape[0]):
                    for j in range(cov_rate_norm.shape[1]):
                        reduced_cov_rate_norm[i, j] = cov_rate_norm[i, j] / (cov_rate_norm[i, i]**0.5 *
                                                                             cov_rate_norm[j, j]**0.5)


                if self.args.debug:

                    plt.clf()

                    plt.subplot(1,3,1)
                    plt.imshow(reduced_cov, origin='lower')
                    plt.colorbar(label="Reduced Covariance")
                    plt.title(f"Reduced Systematic Covariance Matrix for {survey}")
                    plt.xlabel("Redshift Bin")
                    plt.ylabel("Redshift Bin")
                    plt.subplot(1,3,2)
                    plt.imshow(reduced_cov_thresh, origin='lower')
                    plt.colorbar(label="Reduced Covariance")
                    plt.title(f"Reduced Threshold Covariance Matrix for {survey}")
                    plt.xlabel("Redshift Bin")
                    plt.ylabel("Redshift Bin")
                    plt.subplot(1,3,3)
                    plt.imshow(reduced_cov_rate_norm, origin='lower')
                    plt.colorbar(label="Reduced Covariance")
                    plt.title(f"Reduced Rate Norm Covariance Matrix for {survey}")
                    plt.xlabel("Redshift Bin")
                    plt.ylabel("Redshift Bin")
                    plt.savefig(f"cov_sys_{survey}.png")
            else:
                cov_sys = None
            self.fit_args_dict['cov_sys'][survey] = cov_sys

    def calculate_f_norm(self, survey, index):
        """Calculate f_norm, the factor by which the real data is smaller than the simulation,
           for a given survey and dataset index.
        Inputs
        ------
        survey : str
            Name of the survey.
        index : int
            Dataset index.
        """
        z_bins = self.fit_args_dict['z_bins'][survey]
        if self.datasets.get(f"{survey}_DATA_IA_{index}") is not None or self.args.cheat_cc:
            logging.debug("Calculating f_norm using DATA_IA dataset.")
            f_norm = np.sum(self.datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)) / \
                    np.sum(self.datasets[f"{survey}_SIM_IA"].z_counts(z_bins))

        else:

            logging.debug("Couldn't find DATA_IA dataset for f_norm calculation so I am using inferred Ia counts.")
            num_Ia = self.fit_args_dict["n_data"][survey]
            f_norm = np.sum(num_Ia) / \
                    np.sum(self.datasets[f"{survey}_SIM_IA"].z_counts(z_bins))

            # logging.debug("Using all SNe for f_norm calculation.")
            # f_norm = np.sum(self.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)) / \
            #      np.sum(self.datasets[f"{survey}_SIM_ALL"].z_counts(z_bins))

        self.fit_args_dict['f_norm'][survey] = f_norm
        logging.debug(f"Calculated f_norm to be {f_norm}")

    def add_results(self, survey, index=None):
        """ Add results for a given survey and dataset index to the results dictionary to be saved in save_results.
        Inputs
        ------
        survey : str
            Name of the survey.
        index : int
            Dataset index.
        """
        n_datasets = self.fit_args_dict["n_datasets"][survey]
        # This needs to be updated for more parameters later
        if n_datasets > 1:
            survey_name = survey + f"_dataset_{index}"
        else:
            survey_name = survey

        result = self.final_counts[survey]["result"]
        cov = self.final_counts[survey]["covariance"]
        chi = self.final_counts[survey]["chi"]
        z_bins = self.fit_args_dict['z_bins'][survey]

        self.results[survey].append(pd.DataFrame({
            "alpha": result[0],
            "beta": result[1],
            "reduced_chi_squared": chi/(len(z_bins)-2),
            "alpha_error": np.sqrt(cov[0, 0]),
            "beta_error": np.sqrt(cov[1, 1]),
            "cov_alpha_beta": cov[0, 1],
            "survey": survey_name
            }, index=np.array([0])))

    def apply_cuts(self, survey):
        """Apply any cuts specified in the config to the datasets for a given survey.
        The cuts are determined by the CUTS field for each survey in the config file,
        of the form:
        CUTS:
          parameter_name: min_value, max_value      
        For example:
        DES:
          CUTS:
            x1: -3, 3
            c: -0.3, 0.3
        The parameter names much match the column names in the dataset DataFrame exactly.

        Inputs
        ------
        survey : str
            Name of the survey.
        """
        datasets = self.datasets
        with open(self.args.config, 'r') as config_file:
            files_input = yaml.safe_load(config_file)[survey]
        n_datasets = self.fit_args_dict["n_datasets"][survey]

        # Something to think about: Should cuts be applied to ALL datasets, CC datasets and IA datasets,
        # or just IA datasets? For now, I am applying to all datasets.
        cuts = files_input.get("CUTS", None)
        if cuts is not None:
            for col in list(cuts.keys()):
                raw_spec = cuts[col]
                try:
                    parts = [p.strip() for p in str(raw_spec).split(",")]
                    if len(parts) != 2:
                        raise ValueError(
                            f"Expected two comma-separated values for cut, got {len(parts)} part(s)"
                        )
                    min_val = float(parts[0])
                    max_val = float(parts[1])
                except (ValueError, IndexError, TypeError) as exc:
                    logging.error(
                        "Invalid cut specification for survey %s, column %s: %r. "
                        "Expected format 'min,max' with numeric values.",
                        survey,
                        col,
                        raw_spec,
                    )
                    raise ValueError(
                        f"Invalid cut specification for survey '{survey}', column '{col}': "
                        f"{raw_spec!r}. Expected format 'min,max' with numeric values."
                    ) from exc

                logging.info(f"Applying cut on {col} for survey {survey}: min={min_val}, max={max_val}")

                # Apply cuts to all datasets
                if datasets.get(f"{survey}_SIM_ALL") is not None:
                    logging.debug(f"Applying cuts to {survey}_SIM_ALL")
                    before = datasets[f"{survey}_SIM_ALL"].total_counts
                    datasets[f"{survey}_SIM_ALL"].apply_cut(col, min_val, max_val)
                    after = datasets[f"{survey}_SIM_ALL"].total_counts
                    logging.debug(f"Applied cut to {survey}_SIM_ALL: before={before}, after={after} fraction_kept={after/before if before > 0 else 0}")

                if datasets.get(f"{survey}_SIM_IA") is not None:
                    before = datasets[f"{survey}_SIM_IA"].total_counts
                    logging.debug(f"Applying cuts to {survey}_SIM_IA")
                    after = datasets[f"{survey}_SIM_IA"].total_counts
                    logging.debug(f"Applied cut to {survey}_SIM_IA: before={before}, after={after} fraction_kept={after/before if before > 0 else 0}")
                    datasets[f"{survey}_SIM_IA"].apply_cut(col, min_val, max_val)


                if datasets.get(f"{survey}_SIM_CC") is not None:
                    logging.debug(f"Applying cuts to {survey}_SIM_CC")
                    datasets[f"{survey}_SIM_CC"].apply_cut(col, min_val, max_val)

                for i in range(n_datasets):
                    if datasets.get(f"{survey}_DATA_ALL_{i+1}") is not None:
                        before = datasets[f"{survey}_DATA_ALL_{i+1}"].total_counts
                        datasets[f"{survey}_DATA_ALL_{i+1}"].apply_cut(col, min_val, max_val)
                        after = datasets[f"{survey}_DATA_ALL_{i+1}"].total_counts
                        logging.debug(f"Applied cut to {survey}_DATA_ALL_{i+1}: before={before}, after={after} fraction_kept={after/before if before > 0 else 0}")
                    if datasets.get(f"{survey}_DATA_IA_{i+1}") is not None:
                        before = datasets[f"{survey}_DATA_IA_{i+1}"].total_counts
                        datasets[f"{survey}_DATA_IA_{i+1}"].apply_cut(col, min_val, max_val)
                        after = datasets[f"{survey}_DATA_IA_{i+1}"].total_counts
                        logging.debug(f"Applied cut to {survey}_DATA_IA_{i+1}: before={before}, after={after} fraction_kept={after/before if before > 0 else 0}")
                    if datasets.get(f"{survey}_DATA_CC_{i+1}") is not None:
                        logging.debug(f"Applying cuts to {survey}_DATA_CC_{i+1}, from {datasets[f'{survey}_DATA_CC_{i+1}'].total_counts} entries")
                        datasets[f"{survey}_DATA_CC_{i+1}"].apply_cut(col, min_val, max_val)
                        logging.debug(f"After cut, {datasets[f'{survey}_DATA_CC_{i+1}'].total_counts} entries remain")

    def perform_sanity_checks(self, survey):
        """Perform sanity checks on the datasets and fit arguments."""
        logging.debug("Performing sanity checks on datasets")
        # DUMP should be larger than SIM in all bins

        if self.args.plot:
            LaurenNicePlots()
            logging.debug("Generating sanity check plots")

            plt.clf()
            plt.figure(figsize=(8, 6))
            ax1 = plt.subplot(2, 1, 1)
            plt.tight_layout(pad=3.0)
            logging.debug("Starting dump bar 1")
            logging.debug(f"z counts {self.datasets[f'{survey}_DUMP_IA'].z_counts(self.fit_args_dict['z_bins'][survey])}")


            bins = np.linspace(np.min(self.datasets[f"{survey}_DUMP_ALL"].df[self.datasets[f"{survey}_DUMP_ALL"].z_col]),
                               np.max(self.datasets[f"{survey}_DUMP_ALL"].df[self.datasets[f"{survey}_DUMP_ALL"].z_col]), 20)

            labels = ["Uncut Simulation CC", "Uncut Simulation IA", "Simulated Detected IA", "Simulated Detected CC"]
            for i, ds in enumerate([f"{survey}_DUMP_CC", f"{survey}_DUMP_IA", f"{survey}_SIM_IA", f"{survey}_SIM_CC"]):
                data = self.datasets[ds].df
                zcol = self.datasets[ds].z_col
                plt.hist(data[zcol], bins=bins, alpha=1.0, label=labels[i], histtype='step', linewidth=2)

            plt.xlabel("Redshift")
            plt.ylabel("Counts")
            plt.yscale("log")
            plt.legend()
            logging.debug("Generating sanity check plots - part 2")
            plt.subplot(2, 1, 2, sharex=ax1)

            bins = np.linspace(np.min(self.datasets[f"{survey}_DUMP_ALL"].df[self.datasets[f"{survey}_DUMP_ALL"].z_col]),
                               np.max(self.datasets[f"{survey}_DUMP_ALL"].df[self.datasets[f"{survey}_DUMP_ALL"].z_col]), 10)

            labels = ["Uncut Simulation IA+CC", "Simulated Detected IA+CC", f"{survey} Data"]
            for i, ds in enumerate([f"{survey}_DUMP_ALL", f"{survey}_SIM_ALL", f"{survey}_DATA_ALL_1"]):
                data = self.datasets[ds].df
                zcol = self.datasets[ds].z_col
                if "DATA" in ds:
                    plt.hist(data[zcol], bins=bins, alpha=1, label=labels[i], histtype='step', linewidth=2, color = "black")
                else:
                    plt.hist(data[zcol], bins=bins, alpha=1, label=labels[i], histtype='step', linewidth=2)


            plt.xlabel("Redshift")
            plt.yscale("log")
            plt.ylabel("Counts")
            plt.legend()
            logging.debug(f"Saving sanity check plots to sanity_check_counts_{survey}.png ")
            plt.savefig(f"sanity_check_counts_{survey}.png")

        assert self.datasets[f"{survey}_DUMP_ALL"].total_counts >= self.datasets[f"{survey}_SIM_ALL"].total_counts, \
            f"DUMP_ALL dataset has fewer counts than SIM_ALL dataset for survey {survey}!"
        assert all(self.datasets[f'{survey}_DUMP_ALL'].z_counts(self.fit_args_dict['z_bins'][survey]) >=
                     self.datasets[f'{survey}_SIM_ALL'].z_counts(self.fit_args_dict['z_bins'][survey])), \
            f"DUMP_ALL dataset has fewer counts than SIM_ALL dataset in at least one redshift bin for survey {survey}!"
        # No dataset should have zero total counts
        for key in self.datasets.keys():
            assert self.datasets[key].total_counts > 0, f"Dataset {key} has zero total counts!"

        #The ratio between CC and IA should be reasonable
        sim_IA = self.datasets[f"{survey}_SIM_IA"].total_counts
        sim_CC = self.datasets[f"{survey}_SIM_CC"].total_counts
        ratio = sim_CC / sim_IA if sim_IA > 0 else np.inf
        assert ratio < 5, f"Unreasonable CC to IA ratio in SIM datasets for survey {survey}: {ratio}"
        assert ratio > 0.2, f"Unreasonable CC to IA ratio in SIM datasets for survey {survey}: {ratio}"

        dump_IA = self.datasets[f"{survey}_DUMP_IA"].total_counts
        dump_CC = self.datasets[f"{survey}_DUMP_CC"].total_counts
        dump_ratio = dump_CC / dump_IA if dump_IA > 0 else np.inf
        assert dump_ratio < 100, f"Unreasonable CC to IA ratio in DUMP datasets for survey {survey}: {dump_ratio}"
        assert dump_ratio > 0.01, f"Unreasonable CC to IA ratio in DUMP datasets for survey {survey}: {dump_ratio}"

        #np.testing.assert_allclose(dump_ratio, ratio, atol=0.2, err_msg=f"CC to IA ratios in SIM and DUMP datasets differ significantly for survey {survey}: SIM ratio = {ratio}, DUMP ratio = {dump_ratio}")

    # def sanity_check_plot(self, survey, index):
    #     plt.clf()
    #     plt.subplot(1,2,1)

    #     self.z_bins = self.fit_args_dict['z_bins'][survey]

    #     plt.plot(self.datasets[f"{survey}_DUMP_IA"].z_counts(self.z_bins), label='Uncut simulation IA Counts')
    #     plt.plot(self.datasets[f"{survey}_DUMP_CC"].z_counts(self.z_bins), label='Uncut simulation CC Counts')
    #     plt.plot(self.datasets[f"{survey}_DUMP_ALL"].z_counts(self.z_bins), label='Uncut simulation All Counts')

    #     plt.plot(self.datasets[f"{survey}_SIM_IA"].z_counts(self.z_bins), label='Simulated IA Counts')
    #     plt.plot(self.datasets[f"{survey}_SIM_CC"].z_counts(self.z_bins), label='Simulated CC Counts')
    #     plt.plot(self.datasets[f"{survey}_SIM_ALL"].z_counts(self.z_bins), label='Simulated All Counts')
    #     # plt.plot(self.datasets[f"{survey}_SIM_IA"].z_counts(self.z_bins, prob_thresh=PROB_THRESH), ls="--",
    #     #          label='Sim IA Counts Cut')
    #     # plt.plot(self.datasets[f"{survey}_SIM_CC"].z_counts(self.z_bins, prob_thresh=PROB_THRESH), ls="--" ,
    #     #          label='Sim CC Counts Cut')
    #     # plt.plot(self.datasets[f"{survey}_SIM_ALL"].z_counts(self.z_bins, prob_thresh=PROB_THRESH), ls="--",
    #     #          label='Sim All Counts Cut')
    #     plt.yscale("log")
    #     plt.legend()

    #     #bias_cor = self.datasets[f"{survey}_SIM_IA"].z_counts(self.z_bins) / self.datasets[f"{survey}_SIM_ALL"].z_counts(self.z_bins, prob_thresh = 0.5)
    #     #plt.subplot(1,2,2)
    #     #plt.plot(self.datasets[f"{survey}_DATA_IA_{index}"].z_counts(self.z_bins), label='Data IA Counts')
    #     #plt.plot(self.datasets[f"{survey}_DATA_CC_{index}"].z_counts(self.z_bins), label='Data CC Counts')
    #     plt.plot(self.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(self.z_bins), label='Data All Counts')
    #    # plt.plot(self.datasets[f"{survey}_DATA_IA_{index}"].z_counts(self.z_bins, prob_thresh = PROB_THRESH),ls = "--", label='Data IA Counts Cut')
    #    # plt.plot(self.datasets[f"{survey}_DATA_CC_{index}"].z_counts(self.z_bins, prob_thresh = PROB_THRESH),ls = "--", label='Data CC Counts Cut')
    #    # plt.plot(self.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(self.z_bins, prob_thresh = PROB_THRESH)*bias_cor, color = "k", lw= 3,ls = "--", label='Data All Counts Cut w/ BCor')
    #     plt.yscale("log")
    #     plt.legend()
    #     #pulls.extend(list(pull))
    #     plt.savefig(f"{survey}_dataset_{index}_sanity_check.png")