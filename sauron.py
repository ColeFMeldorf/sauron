# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from scipy.stats import binned_statistic as binstat

# Astronomy
from astropy.cosmology import LambdaCDM
cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def main():
    pass


class SN_dataset():
    def __init__(self, dataframe, sntype):
        self.df = dataframe
        self.sntype = sntype
        if self.sntype not in ["IA", "CC", "all"]:
            print(f"unrecognized type: {self.sntype}")
        possible_z_cols = ['zHD', "GENZ"]
        self.z_col = None
        for i in possible_z_cols:
            self.df[i]
            if self.z_col is None:
                self.z_col = i
                print(f"Found z_col {i}")
            else:
                raise ValueError("Multiple valid zcols found")

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
    print(S)

    return S
