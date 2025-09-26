# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library
import yaml

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import minimize, curve_fit, leastsq
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
    args = parser.parse_args()
    files_input = yaml.safe_load(open(args.config, 'r'))

    datasets, surveys, n_datasets = unpack_dataframes(files_input, corecollapse_are_separate)

    results = []
    for survey in surveys:
        z_bins = np.arange(0, 1.4, 0.1)
        N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(z_bins)
        eff_ij = calculate_transfer_matrix(datasets[f"{survey}_DUMP_IA"],  datasets[f"{survey}_SIM_IA"], z_bins)
        for i in range(n_datasets):
            print(f"Working on survey {survey}, dataset {i+1} -------------------")
            # Core Collapse Contamination
            index = i + 1
            n_data = calculate_CC_contamination(datasets, index, survey, z_bins, args.cheat_cc, PROB_THRESH=0.13)

            # This can't stay actually, we can't used DATA_IA because we won't have it irl.
            f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)) / \
                np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            # f_norm = np.sum(n_data) / \
            #     np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            print(f"Calculated f_norm to be {f_norm}")

            fitobj = fit_rate(N_gen=N_gen, f_norm=f_norm, z_bins=z_bins, eff_ij=eff_ij, n_data=n_data)

            results.append(pd.DataFrame({
                "delta_alpha": fitobj.x[0],
                "delta_beta": fitobj.x[1],
                "reduced_chi_squared": fitobj.fun/(len(z_bins) - 2)
                }, index=np.array([0])))

        save_results(results, args)


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


def unpack_dataframes(files_input, corecollapse_are_separate):
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
            if isinstance(survey_dict[file]['PATH'], str):
                paths = survey_dict[file]['PATH']
            if "DATA" in file:
                paths = [survey_dict[file]['PATH']] if isinstance(survey_dict[file]['PATH'], str) \
                    else survey_dict[file]['PATH']
                for i, path in enumerate(paths):
                    datasets[survey+"_"+file+"_"+str(i+1)] = SN_dataset(pd.read_csv(path, comment="#", sep=r"\s+"),
                                                                        sntype, data_name=survey+"_"+file, zcol=zcol)
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
    return datasets, surveys, n_datasets


def calculate_CC_contamination(datasets, index, survey, z_bins, cheat, PROB_THRESH=0.13):

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
        print("S:", S)

        CC_frac = (1 - IA_frac) * S
        IA_frac = 1 - CC_frac
        print("Calculated a Ia frac of:", IA_frac)
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac
    else:
        print("CHEATING AROUND CC")
        IA_frac = np.ones(len(z_bins)-1)
        datasets[f"{survey}_DATA_ALL_{index}"] = datasets[f"{survey}_DATA_IA_{index}"]
        n_data = datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)

    return n_data


def calculate_CC_scale_factor(data_IA, data_CC, sim_IA, sim_CC):
    mu_res = data_IA.mu_res()
    mu_res_CC = data_CC.mu_res()
    mu_res_sim = sim_IA.mu_res()
    mu_res_sim_CC = sim_CC.mu_res()
    N_data = len(mu_res) + len(mu_res_CC)
    n_data = len(mu_res[mu_res > 1]) + len(mu_res_CC[mu_res_CC > 1])
    R = n_data / N_data

    N_IA_sim = len(mu_res_sim)
    n_IA_sim = len(mu_res_sim[mu_res_sim > 1])

    N_CC_sim = len(mu_res_sim_CC)
    n_CC_sim = len(mu_res_sim_CC[mu_res_sim_CC > 1])

    S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)

    return S


def calculate_transfer_matrix(dump, sim, z_bins):
    eff_ij = np.zeros((len(z_bins) - 1, len(z_bins) - 1))
    simulated_events = sim.df
    sim_z_col = sim.z_col

    dump_counts = dump.z_counts(z_bins)

    # This can't be hardcoded!!!!
    num, _, _ = np.histogram2d(simulated_events['SIM_ZCMB'], simulated_events[sim_z_col], bins=[z_bins, z_bins])

    eff_ij = num/dump_counts

    return eff_ij


def chi2(x, N_gen, f_norm, z_bins, eff_ij, n_data):
    alpha, beta = x
    zJ = (z_bins[1:] + z_bins[:-1])/2
    fJ = alpha * (1 + zJ)**beta
    Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=0)
    chi_squared = np.nansum(
        (n_data - Ei)**2 /
        (var_Ei + var_Si)
    )
    return chi_squared

def chi2_nosum(x, N_gen, f_norm, z_bins, eff_ij, n_data):
    alpha, beta = x
    zJ = (z_bins[1:] + z_bins[:-1])/2
    fJ = alpha * (1 + zJ)**beta
    Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=0)
    var_Ei = np.abs(Ei)
    var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=0)
    chi_squared = np.nan_to_num(
        (n_data - Ei)**2 /
        (var_Ei + var_Si)
    )
    return chi_squared

# optimize curve_fit does not allow for passing additional args easily so we define a class
# that can hold the additional args and define the fit function as a method.
class fitClass:
    def __init__(self, N_gen, eff_ij, f_norm):
        self.N_gen = N_gen
        self.eff_ij = eff_ij
        self.f_norm = f_norm

    def fitfun(self, zJ, alpha, beta):
        fJ = alpha * (1 + zJ)**beta
        Ei = np.sum(self.N_gen * self.eff_ij * self.f_norm * fJ, axis=0)
        return Ei


def fit_rate(N_gen=None, f_norm=None, z_bins=None, eff_ij=None, n_data=None):
    # How will this work when I am fitting a non-power law?
    # How do I get the inherent rate in the simulation? Get away from tracking simulated efficiency.
    # Switch to something that returns the covariance matrix.
    fitobj = minimize(chi2, x0=(1, 0), args=(N_gen, f_norm, z_bins, eff_ij, n_data),
                      bounds=[(None, None), (None, None)])
    print("Delta Alpha and Delta Beta:", fitobj.x)
    print("Reduced Chi Squared:", fitobj.fun/(len(z_bins) - 2))
    result, cov_x = leastsq(chi2_nosum, x0=(1.3, 0.3), args=(N_gen, f_norm, z_bins, eff_ij, n_data), full_output=True)[:2]
    cov_x *= np.var(chi2_nosum(result, N_gen, f_norm, z_bins, eff_ij, n_data))
    # inst = fitClass(N_gen, eff_ij, f_norm)
    # zJ = (z_bins[1:] + z_bins[:-1])/2
    # n_data = np.nan_to_num(n_data, nan=0.0)
    # var_Ei = np.abs(Ei)
    # var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=0)
    # sigma=np.sqrt(var_Ei + var_Si)
    # popt, pcov = curve_fit(inst.fitfun, zJ, n_data, absolute_sigma=True, p0=(1, 0), sigma=sigma)
    # print("Curve Fit Results:")
    # print("Delta Alpha and Delta Beta:", popt)
    # print("Covariance Matrix:", pcov)
    print(f"multiplied by var {np.var(chi2_nosum(result, N_gen, f_norm, z_bins, eff_ij, n_data))}")

    print("Delta Alpha and Delta Beta:", result)
    print(f"Standard errors: {np.sqrt(np.diag(cov_x))}")
    print("Covariance Matrix:", cov_x)
    #print("Reduced Chi Squared:", fitobj.fun/(len(z_bins) - 2))

    chi = chi2(np.array([1, 0]), N_gen, f_norm, z_bins, eff_ij, n_data)
    print("Optimal Chi", chi)

    return fitobj


def save_results(results, args):
    for i, result in enumerate(results):
        if i == 0:
            output_df = result
        else:
            output_df = pd.concat([output_df, result], ignore_index=True)

    output_path = args.output
    print(f"Saving to {output_path}")
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
