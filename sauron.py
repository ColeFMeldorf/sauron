# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library
import yaml

import argparse
import glob
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.stats import binned_statistic as binstat


# Astronomy
from astropy.cosmology import LambdaCDM
from astropy.io import fits
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# These need to be added to config file later
corecollapse_are_separate = True



def main():
    parser = argparse.ArgumentParser(description='SAURON: Survey-Agnostic volUmetric Rate Of superNovae')
    parser.add_argument('config', help='Path to the config file (positional argument)')
    parser.add_argument('--output', '-o', default='sauron_output.csv', help='Path to the output file (optional)')
    parser.add_argument("--cheat_cc", action="store_true", help="Cheat and skip CC step. Data_IA will be used as"
                        " Data_All.")
    parser.add_argument("-c", "--covariance", action="store_true", help="Calculate covariance matrix terms.")
    args = parser.parse_args()

    runner = sauron_runner(args)

    datasets, surveys, n_datasets = runner.unpack_dataframes(corecollapse_are_separate)

    for survey in surveys:
        runner.z_bins = np.arange(0, 1.4, 0.1)
        runner.get_counts(survey)
        runner.results = []
        runner.calculate_transfer_matrix(survey)
        PROB_THRESH = 0.13

        # Covariance calculations
        if args.covariance:
            cov_thresh = calculate_covariance_matrix_term(runner.calculate_CC_contamination, [0.05, 0.1, 0.15],
                                                          runner.z_bins, 1, survey)

            rescale_vals = []
            for i in range(100):
                rescale_vals.append(np.random.normal(1, 0.2, size=3))
            cov_rate_norm = calculate_covariance_matrix_term(rescale_CC_for_cov, rescale_vals, runner.z_bins, PROB_THRESH,
                                                             1, survey, datasets, runner.z_bins, False)
            # Hard coding index to one neeeds to change. I don't think this fcn should need index at all. TODO
            cov_sys = cov_thresh + cov_rate_norm
        else:
            cov_sys = None

        for i in range(n_datasets):
            print(f"Working on survey {survey}, dataset {i+1} -------------------")
            # Core Collapse Contamination
            index = i + 1
            n_data = runner.calculate_CC_contamination(PROB_THRESH, index, survey)

            # This can't stay actually, we can't used DATA_IA because we won't have it irl.

            if corecollapse_are_separate:
                f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)) / \
                    np.sum(datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins))
            else:
                f_norm = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(runner.z_bins)) / \
                    np.sum(datasets[f"{survey}_SIM_ALL"].z_counts(runner.z_bins))


            # f_norm = np.sum(n_data) / \
            #     np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            print(f"Calculated f_norm to be {f_norm}")
            alpha, beta, chi, Ei, cov, Ei_err = runner.fit_rate(f_norm=f_norm, n_data=n_data, cov_sys=cov_sys)

            print("Expected Counts:", Ei)

            runner.predicted_counts = Ei
            runner.predicted_counts_err = Ei_err
            runner.n_data = n_data

            runner.results.append(pd.DataFrame({
                "delta_alpha": alpha,
                "delta_beta": beta,
                "reduced_chi_squared": chi/(len(runner.z_bins)-2),
                "alpha_error": np.sqrt(cov[0, 0]),
                "beta_error": np.sqrt(cov[1, 1]),
                "cov_alpha_beta": cov[0, 1],
                }, index=np.array([0])))

        runner.save_results()
        runner.summary_plot()


class SN_dataset():
    def __init__(self, dataframe, sntype, zcol=None, data_name=None, true_z_col=None):
        self.df = dataframe
        self.sntype = sntype
        self._true_z_col = true_z_col
        if self.sntype not in ["IA", "CC", "all"]:
            print(f"unrecognized type: {self.sntype}")

        if zcol is not None:
            possible_z_cols = [zcol]
            z_col_specified = True
        else:
            possible_z_cols = ['zHD', "GENZ", "HOST_ZPHOT"]
            z_col_specified = False

        self.z_col = None
        for i in possible_z_cols:
            if i in self.df.columns:
                if self.z_col is None:
                    self.z_col = i
                else:
                    raise ValueError(f"Multiple valid zcols found in {data_name}. I found: {self.z_col} and {i}")
        if self.z_col is None:
            if z_col_specified:
                raise ValueError(f"Couldn't find specified zcol {zcol} in dataframe for {data_name}!")
            else:
                raise ValueError(f"Couldn't find any valid zcol in dataframe for {data_name}!" \
                    f" I checked: {possible_z_cols}")

        scone_col = []
        for c in self.df.columns:
            if "PROB_SCONE" in c or "SCONE_pred" in c:
                scone_col.append(c)
        if len(scone_col) == 0:
            self.scone_col = None
        elif len(scone_col) > 1:
            raise ValueError(f"Multiple Valid scone columns found in {data_name}! Which do I use? I found: {scone_col}")
        else:
            self.scone_col = scone_col[0]
            print(f"Using scone col {scone_col}")

    @property
    def true_z_col(self):
        return self._true_z_col

    @true_z_col.setter
    def true_z_col(self, value):
        try:
            if value is not None:
                self.df[value]
        except KeyError:
            raise KeyError(f"Couldn't find true z col {value} in dataframe")
        self._true_z_col = value

    def z_counts(self, z_bins, prob_thresh=None):
        if prob_thresh is not None:
            return binstat(self.df[self.z_col][self.prob_scone() > prob_thresh],
                           self.df[self.z_col][self.prob_scone() > prob_thresh], statistic='count', bins=z_bins)[0]

        return binstat(self.df[self.z_col], self.df[self.z_col], statistic='count', bins=z_bins)[0]

    def mu_res(self):
        alpha = 0.146
        beta = 3.03
        mu = 19.416 + self.df.mB + alpha * self.df.x1 - beta * self.df.c
        mu_res = mu-cosmo.distmod(self.df[self.z_col]).value
        return mu_res

    def prob_scone(self):
        if self.scone_col is None:
            raise ValueError("No valid prob_scone column!")
        return self.df[self.scone_col]

    def combine_with(self, dataset, newtype, data_name=None):
        new_df = pd.concat([self.df, dataset.df], join="inner")
        if self.scone_col is not None and dataset.scone_col is not None:
            scone_prob_col = pd.concat([self.prob_scone(), dataset.prob_scone()])
            new_df["PROB_SCONE"] = scone_prob_col
        return SN_dataset(new_df, newtype, zcol=self.z_col, data_name=data_name)
        # Note that this forces the two data sets to have the
        # same z_col. I can't think of a scenario where this would be a problem, but maybe it could be.


class sauron_runner():
    def __init__(self, args):
        self.args = args

    def unpack_dataframes(self, corecollapse_are_separate):
        files_input = yaml.safe_load(open(self.args.config, 'r'))
        surveys = list(files_input.keys())
        datasets = {}
        # Unpack dataframes into SN_dataset objects
        for survey in surveys:
            survey_dict = files_input[survey]
            for i, file in enumerate(list(survey_dict.keys())):
                print(f"Loading {file} for {survey}...")
                sntype = "IA" if "IA" in file else "CC"

                if isinstance(survey_dict[file], dict):
                    zcol = survey_dict[file].get("ZCOL", None)
                else:
                    zcol = None

                # Either use the paths provided or glob the directory provided
                if survey_dict[file].get('PATH') is not None:
                    paths = survey_dict[file]['PATH']
                    paths = [paths] if type(paths) is not list else paths # Make it a list for later
                elif survey_dict[file].get('DIR') is not None:
                    paths = []
                    for dir in survey_dict[file]['DIR']:
                        print(f"Looking in {dir} for files")
                        print(os.listdir(dir))
                        paths.extend(glob.glob(dir + "/**/*.gz"))
                        paths.extend(glob.glob(dir + "*.gz"))  # This extension can't be hardcoded
                    print(f"Found {len(paths)} files in {survey_dict[file]['DIR']}")

                if "DATA" in file:
                    for i, path in enumerate(paths):
                        if ".FITS" in path:
                            dataframe = fits.open(path)[1].data
                            dataframe = pd.DataFrame(np.array(dataframe))
                        elif ".csv" in path:
                            print("Reading csv file")
                            dataframe = pd.read_csv(path, comment="#")
                        else:
                            dataframe = pd.read_csv(path, comment="#", sep=r"\s+")
                        datasets[survey+"_"+file+"_"+str(i+1)] = SN_dataset(dataframe,
                                                                            sntype, data_name=survey+"_"+file,
                                                                            zcol=zcol)
                    n_datasets = len(paths)
                    print("Found", n_datasets, "data sets for", survey)

                else:
                    path = survey_dict[file]['PATH']
                    if ".FITS" in path:
                        dataframe = fits.open(path)[1].data
                        dataframe = pd.DataFrame(np.array(dataframe))
                    elif ".csv" in path:
                        print("Reading csv file")
                        dataframe = pd.read_csv(path, comment="#")
                    else:
                        dataframe = pd.read_csv(path, comment="#", sep=r"\s+")
                    datasets[survey+"_"+file] = SN_dataset(dataframe,
                                                           sntype, data_name=survey+"_"+file, zcol=zcol)

                    datasets[survey+"_"+file].true_z_col = survey_dict[file].get("TRUEZCOL", None)
                    if datasets[survey+"_"+file].true_z_col is None:
                        possible_true_z_cols = ["GENZ", "TRUEZ", "SIMZ", "SIM_ZCMB"]
                        cols_in_df = [col for col in possible_true_z_cols if col in datasets[survey+"_"+file].df.columns]
                        if len(cols_in_df) > 1:
                            raise ValueError(f"Multiple possible true z cols found for {survey}_{file}: {cols_in_df}. "
                                             "Please specify TRUEZCOL in config file.")
                        elif len(cols_in_df) == 1:
                            datasets[survey+"_"+file].true_z_col = cols_in_df[0]
                            print(f"Auto-setting true z col for {survey}_{file} to {cols_in_df[0]}")
                    print(f"Setting true z col for {survey}_{file} to {survey_dict[file].get("TRUEZCOL", None)}")

            # Combine IA and CC files if they are separate
            if corecollapse_are_separate:
                print("Combining IA and CC files..")
                datasets[f"{survey}_DUMP_ALL"] = datasets[f"{survey}_DUMP_IA"].combine_with(
                    datasets[f"{survey}_DUMP_CC"], "all", data_name=survey+"_DUMP_ALL")
                datasets[f"{survey}_SIM_ALL"] = datasets[f"{survey}_SIM_IA"].combine_with(
                    datasets[f"{survey}_SIM_CC"], "all", data_name=survey+"_SIM_ALL")
                for i in range(n_datasets):
                    datasets[f"{survey}_DATA_ALL_{i+1}"] = datasets[f"{survey}_DATA_IA_"+str(i+1)].combine_with(
                        datasets[f"{survey}_DATA_CC_"+str(i+1)], "all", data_name=survey+f"_DATA_ALL_{i+1}")
            # Otherwise, if they aren't seperate, we need to split DUMP and SIM into IA and CC
            else:
                print("Splitting DUMP and SIM files into IA and CC...")
                try:
                    dump_df = datasets[f"{survey}_DUMP_ALL"].df
                    sim_df = datasets[f"{survey}_SIM_ALL"].df
                except KeyError:
                    raise KeyError(f"Couldn't find {survey}_DUMP_ALL or {survey}_SIM_ALL. If your DUMP and SIM files are "
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
                dump_cc_df = dump_df[dump_df[dump_sn_col].isin(ia_vals) == False]
                sim_ia_df = sim_df[sim_df[sim_sn_col].isin(ia_vals_sim)]
                sim_cc_df = sim_df[sim_df[sim_sn_col].isin(ia_vals_sim) == False]

                datasets[f"{survey}_DUMP_IA"] = SN_dataset(dump_ia_df, "IA", zcol=datasets[f"{survey}_DUMP_ALL"].z_col,
                                                           data_name=survey+"_DUMP_IA")
                datasets[f"{survey}_DUMP_CC"] = SN_dataset(dump_cc_df, "CC", zcol=datasets[f"{survey}_DUMP_ALL"].z_col,
                                                           data_name=survey+"_DUMP_CC")
                datasets[f"{survey}_SIM_IA"] = SN_dataset(sim_ia_df, "IA", zcol=datasets[f"{survey}_SIM_ALL"].z_col,
                                                           data_name=survey+"_SIM_IA", true_z_col=datasets[f"{survey}_SIM_ALL"].true_z_col)
                datasets[f"{survey}_SIM_CC"] = SN_dataset(sim_cc_df, "CC", zcol=datasets[f"{survey}_SIM_ALL"].z_col,
                                                           data_name=survey+"_SIM_CC", true_z_col=datasets[f"{survey}_SIM_ALL"].true_z_col)

        self.datasets = datasets
        self.surveys = surveys
        self.n_datasets = n_datasets
        return datasets, surveys, n_datasets

    def get_counts(self, survey):
        z_bins = self.z_bins
        self.N_gen = self.datasets[f"{survey}_DUMP_IA"].z_counts(z_bins)

    def calculate_transfer_matrix(self, survey):
        dump = self.datasets[f"{survey}_DUMP_IA"]
        sim = self.datasets[f"{survey}_SIM_IA"]
        eff_ij = np.zeros((len(self.z_bins) - 1, len(self.z_bins) - 1))

        print("Using true col:", sim.true_z_col, "and recovered col:", sim.z_col)
        simulated_events = sim.df
        sim_z_col = sim.z_col
        true_z_col = sim.true_z_col

        #plt.errorbar(simulated_events[true_z_col], simulated_events[sim_z_col],
        #            yerr=sim.df['REDSHIFT_FINAL_ERR'], fmt='.', alpha=0.1)
        # plt.savefig("transfer_scatter.png")

        dump_counts = dump.z_counts(self.z_bins)

        num, _, _ = np.histogram2d(simulated_events[true_z_col], simulated_events[sim_z_col],
                                   bins=[self.z_bins, self.z_bins])

        eff_ij = num/dump_counts

        self.eff_ij = eff_ij
        return eff_ij

    def fit_rate(self, f_norm=None, n_data=None, cov_sys=0):
        # How will this work when I am fitting a non-power law?
        # How do I get the inherent rate in the simulation? Get away from tracking simulated efficiency.
        z_bins = self.z_bins
        N_gen = self.N_gen
        eff_ij = self.eff_ij
        result, cov_x, infodict = leastsq(chi2, x0=(1, 0), args=(N_gen, f_norm, z_bins, eff_ij, n_data, cov_sys),
                                          full_output=True)[:3]
        N = len(n_data)
        n = len(result)
        cov_x *= (infodict['fvec']**2).sum()/ (N-n)
        # See scipy doc for leastsq for explanation of this covariance rescaling
        print("Delta Alpha and Delta Beta:", result)
        print(f"Standard errors: {np.sqrt(np.diag(cov_x))}")
        print("Covariance Matrix:", cov_x)

        chi = chi2(np.array([1, 0]), N_gen, f_norm, z_bins, eff_ij, n_data, cov_sys=cov_sys)
        print("Optimal Chi", chi)

        fJ = result[0] * (1 + (z_bins[1:] + z_bins[:-1])/2)**result[1]
        print("fJ", np.shape(fJ))
        print("Eff_ij", np.shape(eff_ij))
        print("f_norm", f_norm)
        print("N_gen", np.shape(N_gen))
        Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)

        # Estimate errors on Ei
        alpha_draws = np.random.normal(result[0], np.sqrt(cov_x[0, 0]), size=1000)
        beta_draws = np.random.normal(result[1], np.sqrt(cov_x[1, 1]), size=1000)
        fJ_draws = alpha_draws * (1 + (z_bins[1:] + z_bins[:-1])/2)[:,np.newaxis]**beta_draws
        print("fJ_draws", np.shape(fJ_draws))
        N_gen = N_gen[:, np.newaxis]  # for broadcasting
        eff_ij = np.repeat(eff_ij[:, :, np.newaxis], 1000, axis=2)
        print("N_gen for broadcasting", np.shape(N_gen))
        print("eff_ij for broadcasting", np.shape(N_gen * eff_ij))

        Ei_draws = np.sum(N_gen * eff_ij * f_norm * fJ_draws, axis=0)
        Ei_err = np.std(Ei_draws, axis=1)
        print("Ei_err", Ei_err)

        return result[0], result[1], np.sum(infodict['fvec']), Ei, cov_x, Ei_err

    def save_results(self):
        for i, result in enumerate(self.results):
            if i == 0:
                output_df = result
            else:
                output_df = pd.concat([output_df, result], ignore_index=True)

        output_path = self.args.output
        print(f"Saving to {output_path}")
        output_df.to_csv(output_path, index=False)

    def calculate_CC_contamination(self, PROB_THRESH, index, survey):
        datasets = self.datasets
        z_bins = self.z_bins
        cheat = self.args.cheat_cc

        if not cheat:
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
            print("Calculated a Ia frac of:", IA_frac)
            n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac
        else:
            print("CHEATING AROUND CC")
            IA_frac = np.ones(len(z_bins)-1)
            datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
            n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)

        return n_data

    def summary_plot(self):

        plt.subplot(1, 2, 1)
        z_centers = (self.z_bins[1:] + self.z_bins[:-1]) / 2
        plt.errorbar(z_centers, self.predicted_counts, yerr=self.predicted_counts_err,  fmt='o', label = "Sauron Prediction")
        plt.errorbar(z_centers, self.n_data, yerr=np.sqrt(self.n_data), fmt='o', label = "Data")
        plt.yscale("log")
        plt.legend()
        plt.xlabel("Redshift")
        plt.ylabel("Counts")

        plt.subplot(1, 2, 2)
        plt.imshow(self.eff_ij, extent=(0, np.max(self.z_bins), 0, np.max(self.z_bins)), origin='lower', aspect='auto')
        plt.colorbar()
        plt.title("Efficiency Matrix")
        plt.xlabel("Observed Redshift Bin")
        plt.ylabel("True Redshift Bin")

        plt.savefig("summary_plot.png")


def chi2(x, N_gen, f_norm, z_bins, eff_ij, n_data, cov_sys=0):
    alpha, beta = x
    zJ = (z_bins[1:] + z_bins[:-1])/2
    fJ = alpha * (1 + zJ)**beta
    Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=0)

    cov_stat = np.diag(var_Ei + var_Si)
    if cov_sys is None:
        cov_sys = 0
    cov = cov_stat + cov_sys
    inv_cov = np.linalg.pinv(cov)
    resid_matrix = np.outer(n_data - Ei, n_data - Ei)
    chi_squared = np.sum(inv_cov * resid_matrix, axis=0)

    return chi_squared


def calculate_covariance_matrix_term(sys_func, sys_params, z_bins, *args):
    # Calculate covariance matrix term for a given systematic function and its parameters
    # sys_func should be a function that takes sys_params and returns expected counts
    # args are additional arguments needed for sys_func.
    print("Calculating Covariance Matrix Term...")
    expected_counts = []
    for i, param in enumerate(sys_params):
        expected_counts.append(sys_func(param, *args))

    for i, E in enumerate(expected_counts):
        if i == 0:
            fiducial = E
            C_ij = np.zeros((len(E), len(E)))
        else:
            C_ij_term = np.outer(E - fiducial, E - fiducial) * 1/(len(sys_params)-1) # Fixed this error
            C_ij += C_ij_term

    var = np.diag(C_ij)
    denominator = np.outer(np.sqrt(var), np.sqrt(var))
    denominator[denominator == 0] = 1  # Avoid division by zero

    # plt.subplot(2,1,1)
    # for i in expected_counts:
    #     plt.plot(z_bins[:-1], i)
    # plt.xlabel('Redshift Bin')
    # plt.ylabel('Expected Counts')
    # plt.legend()
    # plt.title("Expected Counts vs Redshift Bins")

    # plt.subplot(2,1,2)
    # plt.imshow(C_ij, extent=(0, np.max(z_bins), 0, np.max(z_bins)), origin='lower', aspect='auto')
    # plt.colorbar()
    # plt.title("Covariance Matrix of Expected Counts")
    # plt.xlabel("Redshift Bin")
    # plt.ylabel("Redshift Bin")
    # plt.savefig("covariance_matrix.png")

    return C_ij


def rescale_CC_for_cov(rescale_vals, PROB_THRESH, index, survey, datasets, z_bins, cheat):
    if cheat:
        IA_frac = np.ones(len(z_bins)-1)
        datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)
    else:
        C_ij = np.zeros((len(z_bins)-1, len(z_bins)-1))
        sim_IA = datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH)

        print("Sim IA counts:", sim_IA)

        sim_CC_df_no_cut = datasets[f"{survey}_SIM_CC"].df
        sim_CC_df = sim_CC_df_no_cut[datasets[f"{survey}_SIM_CC"].prob_scone() > PROB_THRESH]
        types = sim_CC_df_no_cut.TYPE.unique()
        assert(rescale_vals.shape[0] == len(types)), "I got the wrong number of rescale values compared to contamination types" \
            f"({rescale_vals.shape[0]} vs {len(types)})"


        N_CC_sim = np.zeros((len(z_bins)-1, len(types)))
        sim_CC = np.zeros((len(z_bins)-1, len(types)))

        for i,t in enumerate(types):
            rescale_factor = rescale_vals[i]
            #print("Rescale factor:", rescale_factor)
            raw_counts = binstat(sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                sim_CC_df[sim_CC_df.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            sim_CC[:,i] = raw_counts

            raw_counts = binstat(sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                sim_CC_df_no_cut[sim_CC_df_no_cut.TYPE == t][datasets[f"{survey}_SIM_CC"].z_col],
                                statistic='count', bins=z_bins)[0]
            raw_counts = raw_counts * rescale_factor
            N_CC_sim[:,i] = raw_counts
            #print(f"Type {t} counts (rescaled):", raw_counts)

        sim_CC = np.sum(sim_CC, axis = 1)
        print("sim_CC:", sim_CC)


        IA_frac = np.nan_to_num(sim_IA / (sim_IA + sim_CC))
        print("Ia frac denom:", sim_IA + sim_CC)
        print("IA_frac:", IA_frac)
        N_CC_sim = np.sum(N_CC_sim)
        n_CC_sim = np.sum(sim_CC)

        N_Ia_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)
        N_CC_data = datasets[f"{survey}_DATA_CC_{index}"].z_counts(z_bins)

        n_Ia_data = datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
        n_CC_data = datasets[f"{survey}_DATA_CC_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)


        N_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins))
        n_data = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH))
        R = n_data / N_data
        print("R:", R)

        N_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
        n_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH))

        S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)
        print("S:", S)

        CC_frac = (1 - IA_frac) * S
        print("Calculated a CC frac of:", CC_frac)
        IA_frac = np.nan_to_num(1 - CC_frac)
        print("Calculated a Ia frac of:", IA_frac)

        n_true_Ia = datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
        n_true_CC = datasets[f"{survey}_DATA_CC_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH)
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac

    return n_data


if __name__ == "__main__":
    main()
