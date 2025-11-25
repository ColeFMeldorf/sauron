# Standard Library
import pandas as pd
from scipy.stats import binned_statistic as binstat
import logging

# Astronomy
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


class SN_dataset():
    def __init__(self, dataframe, sntype, zcol=None, data_name=None, true_z_col=None):
        self.df = dataframe
        self.sntype = sntype
        self._true_z_col = true_z_col
        if self.sntype not in ["IA", "CC", "all"]:
            logging.warning(f"unrecognized type: {self.sntype}")

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
            for c in self.df.columns:
                logging.debug(f"Available column: {c}")
            if z_col_specified:
                raise ValueError(f"Couldn't find specified zcol {zcol} in dataframe for {data_name}!")
            else:

                raise ValueError(f"Couldn't find any valid zcol in dataframe for {data_name}!"
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
        logging.debug("Determined scone col: " + str(self.scone_col) + f" for dataset {data_name}")

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
            raise ValueError(f"No valid prob_scone column!")
        return self.df[self.scone_col]

    def combine_with(self, dataset, newtype, data_name=None):
        new_df = pd.concat([self.df, dataset.df], join="inner")
        if self.scone_col is not None and dataset.scone_col is not None:
            scone_prob_col = pd.concat([self.prob_scone(), dataset.prob_scone()])
            new_df["PROB_SCONE"] = scone_prob_col
        return SN_dataset(new_df, newtype, zcol=self.z_col, data_name=data_name)
        # Note that this forces the two data sets to have the
        # same z_col. I can't think of a scenario where this would be a problem, but maybe it could be.
