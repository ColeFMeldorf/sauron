# Standard Library
import yaml

import glob
import logging
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.sparse import block_diag
from scipy import stats


# Astronomy
from astropy.cosmology import LambdaCDM
from astropy.io import fits

# Sauron modules
from funcs import (power_law, turnover_power_law, chi2, calculate_covariance_matrix_term, rescale_CC_for_cov,
                   calculate_null_counts)
from SN_dataset import SN_dataset

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
        logging.debug(f"options {args_dict} for survey {survey}")
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
        logging.debug(f"Z BINS {self.fit_args_dict['z_bins'][survey]}")
        self.fit_args_dict["z_centers"][survey] = (self.fit_args_dict["z_bins"][survey][1:] +
                                                   self.fit_args_dict["z_bins"][survey][:-1]) / 2

        simulated_rate_func = args_dict.get("RATE_FUNC", None)
        if simulated_rate_func is not None:
            self.fit_args_dict["simulated_rate_function"][survey] = simulated_rate_func
        else:
            raise ValueError(f"RATE_FUNC must be specified in FIT_OPTIONS for {survey}.")

        simulated_rate_params = args_dict.get("RATE_PARAMS", None)
        if simulated_rate_params is not None:
            simulated_rate_params = [float(i) for i in simulated_rate_params.split(",")]
            self.fit_args_dict["rate_params"][survey] = simulated_rate_params
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
            logging.debug(f"Survey fit options: {fit_args_dict}")
            self.parse_survey_fit_options(fit_args_dict, survey)
            for i, file in enumerate(list(survey_dict.keys())):
                if "DUMP" not in file and "SIM" not in file and "DATA" not in file:
                    continue  # Skip non-data files

                logging.info(f"Loading {file} for {survey}...")
                sntype = "IA" if "IA" in file else "CC"

                if isinstance(survey_dict[file], dict):
                    zcol = survey_dict[file].get("ZCOL", None)
                else:
                    zcol = None

                # Either use the paths provided or glob the directory provided
                if survey_dict[file].get('PATH') is not None:
                    paths = survey_dict[file]['PATH']
                    paths = [paths] if type(paths) is not list else paths  # Make it a list for later
                    paths = glob.glob(paths[0]) if len(paths) == 1 else paths  # Check to see if they meant to glob
                elif survey_dict[file].get('DIR') is not None:
                    paths = []
                    for dir in survey_dict[file]['DIR']:
                        paths.extend(glob.glob(dir + "/**/*.gz"))
                        paths.extend(glob.glob(dir + "*.gz"))  # This extension can't be hardcoded
                    logging.info(f"Found {len(paths)} files in {survey_dict[file]['DIR']}")

                if "DATA" in file:
                    for i, path in enumerate(paths):
                        if ".FITS" in path:
                            dataframe = fits.open(path)[1].data
                            dataframe = pd.DataFrame(np.array(dataframe))
                        elif ".csv" in path:
                            dataframe = pd.read_csv(path, comment="#")
                        else:
                            dataframe = pd.read_csv(path, comment="#", sep=r"\s+")
                        datasets[survey+"_"+file+"_"+str(i+1)] = SN_dataset(dataframe,
                                                                            sntype, data_name=survey+"_"+file,
                                                                            zcol=zcol)
                    n_datasets = len(paths)
                    self.fit_args_dict["n_datasets"][survey] = n_datasets
                    logging.info(f"Found {n_datasets} data sets for {survey}")

                else:
                    dataframe = pd.DataFrame()
                    for path in paths:
                        if ".FITS" in path:
                            dataframe = pd.concat([dataframe, pd.DataFrame(np.array(fits.open(path)[1].data))])
                        elif ".csv" in path:
                            dataframe = pd.concat([dataframe, pd.read_csv(path, comment="#")])
                        else:
                            dataframe = pd.concat([dataframe, pd.read_csv(path, comment="#", sep=r"\s+")])
                    datasets[survey+"_"+file] = SN_dataset(dataframe,
                                                           sntype, data_name=survey+"_"+file, zcol=zcol)

                    datasets[survey+"_"+file].true_z_col = survey_dict[file].get("TRUEZCOL", None)
                    if datasets[survey+"_"+file].true_z_col is None:
                        possible_true_z_cols = ["GENZ", "TRUEZ", "SIMZ", "SIM_ZCMB"]
                        cols_in_df = [col for col in possible_true_z_cols if
                                      col in datasets[survey+"_"+file].df.columns]
                        if len(cols_in_df) > 1:
                            raise ValueError(f"Multiple possible true z cols found for {survey}_{file}: {cols_in_df}. "
                                             "Please specify TRUEZCOL in config file.")
                        elif len(cols_in_df) == 1:
                            datasets[survey+"_"+file].true_z_col = cols_in_df[0]
                            logging.info(f"Auto-setting true z col for {survey}_{file} to {cols_in_df[0]}")
                    logging.info(f"Setting true z col for {survey}_{file} to {survey_dict[file].get("TRUEZCOL", None)}")

            if self.fit_args_dict["cc_are_sep"].get(survey) is None:
                self.fit_args_dict["cc_are_sep"][survey] = True
                logging.warning("CC_ARE_SEPARATE not specified in config file for "f"{survey}. Defaulting to True.")
            # Combine IA and CC files if they are separate
            if self.fit_args_dict["cc_are_sep"][survey]:
                if not self.args.cheat_cc and datasets.get(f"{survey}_DUMP_CC") is not None:
                    logging.info("Combining IA and CC files..")
                    datasets[f"{survey}_DUMP_ALL"] = datasets[f"{survey}_DUMP_IA"].combine_with(
                        datasets[f"{survey}_DUMP_CC"], "all", data_name=survey+"_DUMP_ALL")
                    datasets[f"{survey}_SIM_ALL"] = datasets[f"{survey}_SIM_IA"].combine_with(
                        datasets[f"{survey}_SIM_CC"], "all", data_name=survey+"_SIM_ALL")
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
                logging.info("Splitting DUMP and SIM files into IA and CC...")
                try:
                    dump_df = datasets[f"{survey}_DUMP_ALL"].df
                    sim_df = datasets[f"{survey}_SIM_ALL"].df
                except KeyError:
                    raise KeyError(f"Couldn't find {survey}_DUMP_ALL or {survey}_SIM_ALL."
                                   " If your DUMP and SIM files are "
                                   "separate for IA and CC, set corecollapse_are_separate to True.")

                try:
                    dump_sn_col = survey_dict["DUMP_ALL"]["SNTYPECOL"]
                    ia_vals = survey_dict["DUMP_ALL"]["IA_VALS"]
                    sim_sn_col = survey_dict["SIM_ALL"]["SNTYPECOL"]
                    ia_vals_sim = survey_dict["SIM_ALL"]["IA_VALS"]
                except KeyError:
                    raise KeyError(f"Couldn't find SNTYPECOL or IA_VALS in config for {survey}. These are needed to "
                                   "separate DUMP and SIM into IA and CC.")

                dump_ia_df = dump_df[dump_df[dump_sn_col].isin(ia_vals)]
                dump_cc_df = dump_df[~dump_df[dump_sn_col].isin(ia_vals)]
                sim_ia_df = sim_df[sim_df[sim_sn_col].isin(ia_vals_sim)]
                sim_cc_df = sim_df[~sim_df[sim_sn_col].isin(ia_vals_sim)]

                datasets[f"{survey}_DUMP_IA"] = SN_dataset(dump_ia_df, "IA", zcol=datasets[f"{survey}_DUMP_ALL"].z_col,
                                                           data_name=survey+"_DUMP_IA")
                datasets[f"{survey}_DUMP_CC"] = SN_dataset(dump_cc_df, "CC", zcol=datasets[f"{survey}_DUMP_ALL"].z_col,
                                                           data_name=survey+"_DUMP_CC")
                datasets[f"{survey}_SIM_IA"] = SN_dataset(sim_ia_df, "IA", zcol=datasets[f"{survey}_SIM_ALL"].z_col,
                                                          data_name=survey+"_SIM_IA",
                                                          true_z_col=datasets[f"{survey}_SIM_ALL"].true_z_col)
                datasets[f"{survey}_SIM_CC"] = SN_dataset(sim_cc_df, "CC", zcol=datasets[f"{survey}_SIM_ALL"].z_col,
                                                          data_name=survey+"_SIM_CC",
                                                          true_z_col=datasets[f"{survey}_SIM_ALL"].true_z_col)

        self.datasets = datasets
        self.surveys = surveys

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

        hist_true = np.histogram(simulated_events[true_z_col], bins=z_bins)
        hist_sim = np.histogram(simulated_events[sim_z_col], bins=z_bins)

        logging.debug(f"true z col bins: {hist_true}")
        logging.debug(f"sim z col bins: {hist_sim}")
        logging.debug(f"total unbinned simulated events: {len(simulated_events)}")
        logging.debug(f"Total in truez {hist_true[0].sum()}")
        logging.debug(f"Total in simz {hist_sim[0].sum()}")

        z_bins_expanded = np.concatenate(([-np.inf], z_bins, [np.inf]))

        num, _, _ = np.histogram2d(simulated_events[true_z_col], simulated_events[sim_z_col],
                                   bins=[z_bins_expanded, z_bins])

        logging.debug(np.sum(num))
        logging.debug(num)

        if np.any(dump_counts == 0):
            logging.warning("Some redshift bins have zero simulated events! This may cause issues.")
            bad_bins = np.where(dump_counts == 0)[0]
            upper_bad_bins = bad_bins + 1
            unique_bad_bins = np.sort(np.unique(np.concatenate((bad_bins, upper_bad_bins))))
            logging.warning("Specifically, these are the bin edges of the zero count bins:", z_bins[unique_bad_bins])

        eff_ij = num/dump_counts
        logging.debug(f"effij: {eff_ij}")

        self.fit_args_dict['eff_ij'][survey] = eff_ij

        logging.debug(f"sum of eff_ij along axis 0: {np.sum(eff_ij, axis=0)}")
        logging.debug(f"sum of eff_ij along axis 1: {np.sum(eff_ij, axis=1)}")

        logging.debug(f"eff * dump counts: {np.sum(eff_ij * dump_counts, axis=0)}")

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
            if c is None:
                cov_sys_list[i] = np.zeros((len(z_centers), len(z_centers)))
        cov_sys = block_diag(cov_sys_list).toarray()

        logging.debug("Shapes:")
        logging.debug(f"z_centers: {z_centers.shape}")
        logging.debug(f"f_norm: {f_norms.shape}")
        logging.debug(f"n_data: {n_data.shape}")
        logging.debug(f"N_gen: {N_gen.shape}")
        logging.debug(f"eff_ij: {eff_ij.shape}")
        logging.debug(f"cov_sys: {cov_sys.shape}")

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
        # The above are only really needed for debugging.

        logging.warning("This doesn't work for multiple surveys yet!")

        true_rate_function = func_name_dictionary.get(self.fit_args_dict["simulated_rate_function"][survey])
        logging.debug(f"Using true rate function: {true_rate_function}")
        logging.debug(f"With params: {self.fit_args_dict['rate_params'][survey]}")
        null_counts = calculate_null_counts(z_bins_list, z_centers, N_gen, cosmo=cosmo,
                                            true_rate_function=true_rate_function,
                                            rate_params=self.fit_args_dict["rate_params"][survey])

        result, cov_x, infodict = leastsq(chi2, x0=self.x0, args=(null_counts, f_norms, z_centers, eff_ij,
                                          n_data, self.rate_function, cov_sys),
                                          full_output=True)[:3]
        logging.debug(f"Least Squares Result: {result}")
        N = len(n_data)
        n = len(result)
        cov_x *= (infodict['fvec']**2).sum() / (N-n)
        # See scipy doc for leastsq for explanation of this covariance rescaling
        logging.debug(f"Standard errors: {np.sqrt(np.diag(cov_x))}")

        fJ = result[0] * (1 + z_centers)**result[1]
        Ei = np.sum(N_gen * eff_ij * f_norms * fJ, axis=0)

        logging.debug(f"Predicted Counts Ei: {Ei}")
        fJ_0 = self.x0[0] * (1 + z_centers)**self.x0[1]
        x0_counts = np.sum(null_counts * eff_ij * f_norms * fJ_0, axis=0)
        logging.debug(f"Counts with x0: {x0_counts}")

        # Estimate errors on Ei
        alpha_draws = np.random.normal(result[0], np.sqrt(cov_x[0, 0]), size=1000)
        beta_draws = np.random.normal(result[1], np.sqrt(cov_x[1, 1]), size=1000)
        fJ_draws = alpha_draws * (1 + z_centers)[:, np.newaxis]**beta_draws
        N_gen = N_gen[:, np.newaxis]  # for broadcasting
        eff_ij = np.repeat(eff_ij[:, :, np.newaxis], 1000, axis=2)
        f_norms = np.atleast_1d(f_norms)
        f_norms = f_norms[:, np.newaxis]  # for broadcasting

        # I am not confident this is correct. Check later.

        Ei_draws = np.sum(N_gen * eff_ij * f_norms * fJ_draws, axis=0)
        Ei_err = np.std(Ei_draws, axis=1)

        self.final_counts[survey]["predicted_counts"] = Ei
        self.final_counts[survey]["observed_counts"] = n_data
        self.final_counts[survey]["predicted_counts_err"] = Ei_err
        self.n_data = n_data

        self.final_counts[survey]["result"] = result
        self.final_counts[survey]["covariance"] = cov_x
        self.final_counts[survey]["chi"] = np.sum(infodict['fvec'])
        # Should there not be an index here?

        return result, np.sum(infodict['fvec']), Ei, cov_x, Ei_err

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

    def calculate_CC_contamination(self, PROB_THRESH, index, survey):
        """Calculate CC contamination for a given survey and dataset index.

        Inputs
        ------
        PROB_THRESH : float
           Probability threshold for classification, this should be the probability the SN is believed to be IA.
        index : int
            Dataset index.
        survey : str
            Name of the survey.
        """
        datasets = self.datasets
        z_bins = self.fit_args_dict['z_bins'][survey]
        cheat = self.args.cheat_cc

        if not cheat and datasets.get(f"{survey}_DUMP_CC") is not None:
            IA_frac = (datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH) /
                       datasets[f"{survey}_SIM_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH))
            N_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins))
            n_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH))
            R = n_data / N_data

            N_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            n_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH))

            N_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins))
            n_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins, prob_thresh=PROB_THRESH))

            S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)

            CC_frac = (1 - IA_frac) * S
            IA_frac = np.nan_to_num(1 - CC_frac)
            logging.debug(f"Calculated a Ia frac of: {IA_frac}")
            n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac
        else:
            logging.warning("SKIPPING CC CONTAMINATION STEP. USING DATA_IA AS DATA_ALL.")
            datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
            n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)

        self.fit_args_dict["n_data"][survey] = n_data
        return n_data

    def generate_chi2_map(self, survey):
        """Generate an array of chi2 values over a grid of alpha and beta values for a given survey.
        For now, this only works for the power law fit function.
        Inputs
        ------
        survey : str
            Name of the survey.
        """

        fit_args_dict = self.fit_args_dict
        chi2_map = np.empty((50, 50))
        logging.info(f"Generating chi2 map for survey: {survey}")
        logging.info(fit_args_dict['z_bins'][survey])
        z_centers = self.fit_args_dict['z_centers'][survey]

        if len(fit_args_dict['N_gen'][survey]) != len(z_centers):
            num_surveys = len(fit_args_dict['N_gen'][survey]) / len(z_centers)
            assert (num_surveys % 1 == 0), "N_gen length is not a multiple of z_centers length!"
            z_centers = np.tile(z_centers, int(num_surveys))
            logging.debug("updated z_centers for chi2 map:", z_centers)

        for i, a in enumerate(np.linspace(0.7, 1.3, 50)):
            for j, b in enumerate(np.linspace(-0.3, 0.3, 50)):

                values = (a, b)

                chi2_result = chi2(values, fit_args_dict['N_gen'][survey], fit_args_dict['f_norm'][survey],
                                   z_centers,
                                   fit_args_dict['eff_ij'][survey],
                                   fit_args_dict['n_data'][survey],
                                   self.rate_function,
                                   fit_args_dict['cov_sys'][survey])
                chi2_map[i][j] = np.sum(chi2_result**2)
        return chi2_map

    def summary_plot(self):
        """ Generate a plot of the results, including predicted vs observed counts and chi2 contours."""
        logging.info(f"Results: {self.results}")
        surveys = self.results.keys()
        logging.info("Generating summary plots for surveys:", list(surveys))
        num_plots = len(surveys) + 1
        sides = int(np.ceil(num_plots/2))
        fig, ax = plt.subplots(2, sides, figsize=(12, 6))
        ax = ax.flatten()
        for i, survey in enumerate(surveys):
            logging.info(f"Generating summary plot for {survey}")
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
                ax1.legend()
                ax1.set_xlabel("Redshift")
                ax1.set_ylabel("Counts")
                ax1.set_yscale("log")

            ax2 = ax[i+1]
            chi2_map = self.generate_chi2_map(s)
            normalized_map = chi2_map - np.min(chi2_map) + 0.0001
            # plt.subplot(1, len(surveys), i + 1)
            # plt.imshow(normalized_map, extent=(-0.1, 0.1, 0.9, 1.1), origin='lower', aspect='auto', cmap="jet")
            ax2.axvline(0, color='black', linestyle='--')
            ax2.axhline(1, color='black', linestyle='--')
            from scipy.stats import chi2 as chi2_scipy

            if isinstance(self.results[s], list):
                df = self.results[s][0]
            else:
                df = self.results[s]

            a = np.mean(df["alpha_error"]**2)
            b = np.mean(df["beta_error"]**2)
            c = np.mean(df["cov_alpha_beta"])
            cov = np.array([[a, c], [c, b]])
            sigma_1 = chi2_scipy.ppf([0.68], 2)
            sigma_2 = chi2_scipy.ppf([0.95], 2)
            norm = np.sqrt((2 * np.pi) ** 2 * np.linalg.det(cov))

            sigma_1_exp = np.exp((-1/2) * sigma_1)
            sigma_1_exp = sigma_1_exp[0] / norm
            sigma_2_exp = np.exp((-1/2) * sigma_2)
            sigma_2_exp = sigma_2_exp[0] / norm
            y = np.linspace(0.7, 1.3, 50)
            x = np.linspace(-0.3, 0.3, 50)
            x, y = np.meshgrid(x, y)
            CS = ax2.contour(x, y, normalized_map, levels=[sigma_2_exp, sigma_1_exp], colors="C" + str(i+1))
            # label the contours by survey
            fmt = {}
            strs = [f'1 sigma {survey}', f'2 sigma {survey}']
            for k, label_str in zip(CS.levels, strs):
                fmt[k] = label_str
            ax2.clabel(CS, CS.levels, fmt=fmt, fontsize=10)
            ax2.legend()

        fig.savefig("summary_plot.png")

    def calculate_covariance(self, PROB_THRESH=0.13):
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
                cov_thresh = calculate_covariance_matrix_term(self.calculate_CC_contamination, [0.05, 0.1, 0.15],
                                                              self.fit_args_dict["z_bins"][survey], 1, survey)

                xx = np.linspace(0.01, 0.99, 10)
                X = stats.norm(loc=1, scale=0.2)
                vals = X.ppf(xx)
                grid = np.meshgrid(vals, vals, vals, indexing='ij')
                grid0 = grid[0].flatten()
                grid1 = grid[1].flatten()
                grid2 = grid[2].flatten()
                rescale_vals = np.array([grid0, grid1, grid2]).T

                cov_rate_norm = calculate_covariance_matrix_term(rescale_CC_for_cov, rescale_vals,
                                                                 self.fit_args_dict["z_bins"][survey], PROB_THRESH,
                                                                 1, survey, self.datasets,
                                                                 self.fit_args_dict["z_bins"][survey], False)
                # Hard coding index to one needs to change. TODO: Refactor to avoid hardcoded index value
                #  (currently set to 1). This function should not need index at all.
                cov_sys = cov_thresh + cov_rate_norm
                logging.debug(f"Cov sys shape in calc cov: {cov_sys.shape}")
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
        if self.fit_args_dict["cc_are_sep"][survey]:
            f_norm = np.sum(self.datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)) / \
                    np.sum(self.datasets[f"{survey}_SIM_IA"].z_counts(z_bins))

        else:
            f_norm = np.sum(self.datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)) / \
                np.sum(self.datasets[f"{survey}_SIM_ALL"].z_counts(z_bins))

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
