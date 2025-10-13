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
            # Hard coding index to one neeeds to change. I don't think this fcn should need index at all.
            cov_sys = cov_thresh + cov_rate_norm
        else:
            cov_sys = None

        # plt.show()
        # plt.subplot(2, 1, 1)
        # var = np.diag(cov_thresh)
        # denominator = np.outer(np.sqrt(var), np.sqrt(var))
        # denominator[denominator == 0] = 1  # Avoid division by zero
        # plt.imshow(cov_thresh/denominator, extent=(0, np.max(runner.z_bins), 0, np.max(runner.z_bins)),
        #            origin='lower', aspect='auto')
        # plt.colorbar()
        # plt.title("Covariance Matrix from Varying CC Threshold")
        # plt.xlabel("Redshift Bin")
        # plt.ylabel("Redshift Bin")
        # plt.subplot(2, 1, 2)
        # var = np.diag(cov_rate_norm)
        # denominator = np.outer(np.sqrt(var), np.sqrt(var))
        # denominator[denominator == 0] = 1  # Avoid division by zero
        # plt.imshow(cov_rate_norm/denominator, extent=(0, np.max(runner.z_bins), 0, np.max(runner.z_bins)),
        #            origin='lower', aspect='auto')
        # print(cov_rate_norm/denominator)
        # plt.colorbar()
        # plt.title("Covariance Matrix from Varying CC Rate Normalization")
        # plt.xlabel("Redshift Bin")
        # plt.ylabel("Redshift Bin")
        # plt.savefig("covariance_matrices.png")
        # print("Covariance matrices saved to covariance_matrices.png")

        for i in range(n_datasets):
            print(f"Working on survey {survey}, dataset {i+1} -------------------")
            # Core Collapse Contamination
            index = i + 1
            n_data = runner.calculate_CC_contamination(PROB_THRESH, index, survey)

            # This can't stay actually, we can't used DATA_IA because we won't have it irl.
            f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(runner.z_bins)) / \
                np.sum(datasets[f"{survey}_SIM_IA"].z_counts(runner.z_bins))
            # f_norm = np.sum(n_data) / \
            #     np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            print(f"Calculated f_norm to be {f_norm}")
            alpha, beta, chi, Ei, cov = runner.fit_rate(f_norm=f_norm, n_data=n_data, cov_sys=cov_sys)

            print("Expected Counts:", Ei)

            runner.results.append(pd.DataFrame({
                "delta_alpha": alpha,
                "delta_beta": beta,
                "reduced_chi_squared": chi/(len(runner.z_bins)-2),
                "alpha_error": np.sqrt(cov[0, 0]),
                "beta_error": np.sqrt(cov[1, 1]),
                "cov_alpha_beta": cov[0, 1],
                }, index=np.array([0])))

        runner.save_results()


class SN_dataset():
    def __init__(self, dataframe, sntype, zcol=None, data_name=None):
        self.df = dataframe
        self.sntype = sntype
        if self.sntype not in ["IA", "CC", "all"]:
            print(f"unrecognized type: {self.sntype}")

        if zcol is not None:
            possible_z_cols = [zcol]
        else:
            possible_z_cols = ['zHD', "GENZ", "HOST_ZPHOT"]

        self.z_col = None
        for i in possible_z_cols:
            try:
                self.df[i]  # better way to do this?
                if self.z_col is None:
                    self.z_col = i
                    print(f"Found z_col {i}")
                else:
                    raise ValueError(f"Multiple valid zcols found in {data_name}. I found: {self.z_col} and {i}")
            except KeyError:
                pass

        scone_col = []
        for c in self.df.columns:
            if "PROB_SCONE" in c:
                scone_col.append(c)
        if len(scone_col) == 0:
            print(f"No valid prob_scone column in {data_name}!")
            self.scone_col = None
        elif len(scone_col) > 1:
            raise ValueError(f"Multiple Valid scone columns found in {data_name}! Which do I use? I found: {scone_col}")
        else:
            self.scone_col = scone_col[0]
            print(f"Using scone col {scone_col}")

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
                        datasets[survey+"_"+file+"_"+str(i+1)] = SN_dataset(pd.read_csv(path, comment="#", sep=r"\s+"),
                                                                            sntype, data_name=survey+"_"+file,
                                                                            zcol=zcol)
                    n_datasets = len(paths)
                    print("Found", n_datasets, "data sets for", survey)

                else:
                    path = survey_dict[file]['PATH']
                    datasets[survey+"_"+file] = SN_dataset(pd.read_csv(path, comment="#", sep=r"\s+"),
                                                           sntype, data_name=survey+"_"+file, zcol=zcol)

        # Combine IA and CC files if they are separate
        if corecollapse_are_separate:
            print("Combining IA and CC files..")
            for survey in surveys:
                datasets[f"{survey}_DUMP_ALL"] = datasets[f"{survey}_DUMP_IA"].combine_with(
                    datasets[f"{survey}_DUMP_CC"], "all", data_name=survey+"_DUMP_ALL")
                datasets[f"{survey}_SIM_ALL"] = datasets[f"{survey}_SIM_IA"].combine_with(
                    datasets[f"{survey}_SIM_CC"], "all", data_name=survey+"_SIM_ALL")
                for i in range(n_datasets):
                    datasets[f"{survey}_DATA_ALL_{i+1}"] = datasets[f"{survey}_DATA_IA_"+str(i+1)].combine_with(
                        datasets[f"{survey}_DATA_CC_"+str(i+1)], "all", data_name=survey+f"_DATA_ALL_{i+1}")

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
        simulated_events = sim.df
        sim_z_col = sim.z_col

        dump_counts = dump.z_counts(self.z_bins)

        # This can't be hardcoded!!!! -------------------V
        num, _, _ = np.histogram2d(simulated_events['SIM_ZCMB'], simulated_events[sim_z_col],
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
        Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)

        return result[0], result[1], np.sum(infodict['fvec']), Ei, cov_x

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
