# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library
import yaml


import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binned_statistic as binstat


# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# These need to be added to config file later
corecollapse_are_separate = True


def main():
    files_input = yaml.safe_load(open("sauron_config.yaml"))
    surveys = list(files_input.keys())

    datasets = {}

    for survey in surveys:
        survey_dict = files_input[survey]
        for i, file in enumerate(list(survey_dict.keys())):
            sntype = "IA" if "IA" in file else "CC"
            datasets[survey+"_"+file] = SN_dataset(pd.read_csv(survey_dict[file], comment="#", sep=r"\s+"), sntype)

    print(datasets)

    if corecollapse_are_separate:
        print("Combining IA and CC files..")
        for survey in surveys:
            datasets[f"{survey}_DUMP_ALL"] = datasets[f"{survey}_DUMP_IA"].combine_with(
                datasets[f"{survey}_DUMP_CC"], "all")
            datasets[f"{survey}_SIM_ALL"] = datasets[f"{survey}_SIM_IA"].combine_with(
                datasets[f"{survey}_SIM_CC"], "all")
            datasets[f"{survey}_DATA_ALL"] = datasets[f"{survey}_DATA_IA"].combine_with(
                datasets[f"{survey}_DATA_CC"], "all")
        print("Done!")

    for survey in surveys:

        z_bins = np.arange(0, 1, 0.1)

        # Core Collapse Contamination
        PROB_THRESH = 0.13

        IA_frac = (datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH) /
                   datasets[f"{survey}_SIM_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH))
        N_data = np.sum(datasets[f"{survey}_DATA_ALL"].z_counts(z_bins))
        n_data = np.sum(datasets[f"{survey}_DATA_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH))
        R = n_data / N_data

        N_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
        n_IA_sim = np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins, prob_thresh=PROB_THRESH))

        N_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins))
        n_CC_sim = np.sum(datasets[f"{survey}_SIM_CC"].z_counts(z_bins, prob_thresh=PROB_THRESH))

        S = (R * N_IA_sim - n_IA_sim) / (n_CC_sim - R * N_CC_sim)

        CC_frac = (1 - IA_frac) * S
        IA_frac = 1 - CC_frac

        eff_ij = calculate_transfer_matrix(datasets[f"{survey}_DUMP_IA"],  datasets[f"{survey}_SIM_IA"], z_bins)

        N_gen = datasets[f"{survey}_DUMP_IA"].z_counts(z_bins)
        f_norm = 1/50
        n_data = datasets[f"{survey}_DATA_ALL"].z_counts(z_bins, prob_thresh=PROB_THRESH) * IA_frac

        # How will this work when I am fitting a non-power law?
        # How do I get the inherent rate in the simulation? Get away from tracking simulated efficiency.
        fitobj = minimize(chi2, x0=(2, 1), args=(N_gen, f_norm, z_bins, eff_ij, n_data), bounds=[(0, None), (0, None)])

        print(fitobj.x)
        print(fitobj.fun/(len(z_bins) - 2))


class SN_dataset():
    def __init__(self, dataframe, sntype):
        self.df = dataframe
        self.sntype = sntype
        if self.sntype not in ["IA", "CC", "all"]:
            print(f"unrecognized type: {self.sntype}")
        possible_z_cols = ['zHD', "GENZ", "HOST_ZPHOT"]
        self.z_col = None
        for i in possible_z_cols:
            try:
                self.df[i]
                if self.z_col is None:
                    self.z_col = i
                    print(f"Found z_col {i}")
                else:
                    raise ValueError("Multiple valid zcols found")
            except KeyError:
                pass

        scone_col = []
        for c in self.df.columns:
            if "PROB_SCONE" in c:
                scone_col.append(c)
        if len(scone_col) == 0:
            print("No valid prob_scone column!")
            self.scone_col = None
        elif len(scone_col) > 1:
            raise ValueError(f"Multiple Valid scone columns found! Which do I use? I found: {scone_col}")
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

    def combine_with(self, dataset, newtype):
        new_df = pd.concat([self.df, dataset.df], join="inner")
        if self.scone_col is not None and dataset.scone_col is not None:
            scone_prob_col = pd.concat([self.prob_scone(), dataset.prob_scone()])
            new_df["PROB_SCONE"] = scone_prob_col
        return SN_dataset(new_df, newtype)


def calculate_transfer_matrix(dump, sim, z_bins):

    eff_ij = np.zeros((len(z_bins) - 1, len(z_bins) - 1))
    dumped_events = dump.df
    simulated_events = sim.df
    dump_z_col = dump.z_col
    sim_z_col = sim.z_col

    for i in range(len(z_bins) - 1):
        dump_events_subset = dumped_events[(dumped_events[dump_z_col] > z_bins[i])
                                           & (dumped_events[dump_z_col] < z_bins[i+1])]
        simulated_events_subset = simulated_events[(simulated_events[sim_z_col] > z_bins[i])
                                                   & (simulated_events[sim_z_col] < z_bins[i+1])]
        dump_counts_subset = binstat(dump_events_subset[dump_z_col],
                                     dump_events_subset[dump_z_col], statistic='count', bins=z_bins)[0]
        simulated_counts_subset = binstat(simulated_events_subset[sim_z_col],
                                          simulated_events_subset[sim_z_col], statistic='count', bins=z_bins)[0]
        eff_ij[i, :] = simulated_counts_subset / dump_counts_subset
        eff_ij[i, :][np.where(dump_counts_subset == 0)] = 0

    return eff_ij


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


def chi2(x, N_gen, f_norm, z_bins, eff_ij, n_data):
    alpha, beta = x
    zJ = (z_bins[1:] + z_bins[:-1])/2
    fJ = alpha * (1 + zJ)**beta
    Ei = np.sum(N_gen * eff_ij * f_norm * fJ, axis=1)
    var_Ei = Ei
    var_Si = np.sum(N_gen * eff_ij * f_norm**2 * fJ**2, axis=1)
    chi_squared = np.sum(
        (n_data - Ei)**2 /
        (var_Ei + var_Si)
    )
    return chi_squared


if __name__ == "__main__":
    main()
