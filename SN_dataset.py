# Standard Library
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic as binstat
import logging
logger = logging.getLogger(__name__)
# Astronomy
from astropy.cosmology import LambdaCDM
from astropy.io import fits

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


class SN_dataset():
    """A class to hold a dataset of supernovae for one survey and type."""
    def __init__(self, paths, sntype, data_name, zcol=None, true_z_col=None, sntypecol=None, survey_dict=None):
        self.data_name = data_name
        logging.debug(f"Initializing SN_dataset for {data_name} with paths: {paths}")
        if not isinstance(paths, list):
            paths = [paths]

        #Survey dict represents all the information valid for 1 survey. Z_bins, cuts, etc.
        #Dataset dict represents all the information valid for 1 of the datasets within that survey.
        # For instance, the redshift column name for that file.
        if survey_dict is not None:
            # Either DUMP, SIM, or DATA should be in the data name. Extract the correct one.
            possible_prefixes = ["DUMP", "SIM", "DATA"]
            possible_extensions = ["_ALL", "_IA", "_CC"]
            possible_types = [f"{p}{e}" for p in possible_prefixes for e in possible_extensions]
            type_in_name = [t for t in possible_types if t in data_name]
            if len(type_in_name) == 0:
                raise ValueError(f"Couldn't find any of {possible_types} in data_name {data_name} to determine dataset type!")
            elif len(type_in_name) > 1:
                raise ValueError(f"Found multiple of {possible_types} in data_name {data_name}! Can't determine dataset type. Found: {type_in_name}")
            else:
                type_in_name = type_in_name[0]

            # Now that we have the type, determine the corresponding dataset dict.
            corresponding_key = [k for k in survey_dict.keys() if type_in_name in k]
            if len(corresponding_key) == 0:
                raise ValueError(f"Couldn't find a key in survey_dict with {type_in_name} in it for data_name {data_name}! Survey dict keys: {survey_dict.keys()}")
            elif len(corresponding_key) > 1:
                raise ValueError(f"Found multiple keys in survey_dict with {type_in_name} in it for data_name {data_name}! Can't determine which to use. Found: {corresponding_key}")
            else:
                corresponding_key = corresponding_key[0]

            dataset_dict = survey_dict[corresponding_key]
        else:
            dataset_dict = None

        if zcol is not None and dataset_dict is not None:
            raise ValueError("Cannot specify zcol both in config file and as a column in the dataset! Please choose one or the other.")
        elif dataset_dict is not None:
            zcol = dataset_dict.get("ZCOL", None)

        if true_z_col is not None and dataset_dict is not None:
            raise ValueError("Cannot specify true_z_col both in config file and as a column in the dataset! Please choose one or the other.")
        elif dataset_dict is not None:
            true_z_col = dataset_dict.get("TRUEZCOL", None)

        if sntypecol is not None and dataset_dict is not None:
            raise ValueError("Cannot specify sntypecol both in config file and as a column in the dataset! Please choose one or the other.")
        elif dataset_dict is not None:
            sntypecol = dataset_dict.get("SNTYPECOL", None)

        cuts = survey_dict.get("CUTS", None) if survey_dict is not None else None
        subsets = survey_dict.get("SUBSET", None) if survey_dict is not None else None

        # Initialize these here so they exist. If z_col fails because it is None, it is likely due to this
        # never being set.
        self.z_col = None
        self.scone_col = None
        self._true_z_col = None

        dataframe = pd.DataFrame()
        for path in paths:
            if path is None:
                continue
            logging.debug(f"Setting self._true_z_col to {true_z_col} in dataset {data_name}")
            self._true_z_col = true_z_col
            submitted_subset_cols = subsets.keys() if subsets is not None else None
            self.actual_subset_cols = self.determine_needed_columns(path, zcol, subset_cols=submitted_subset_cols)
            # This is the actual subset cols determined for this file. It may be different from the requested subset
            # cols if the names don't match.
            potential_cols = self.actual_subset_cols.copy() if self.actual_subset_cols is not None else []
             # Start with the subset cols, since those are definitely needed if they exist.
            potential_cols.extend([self.z_col, self.scone_col, self._true_z_col]) # Include redshift and scone columns

            if "DUMP" not in data_name: # If this is a dump dataset, cuts aren't applied.
                potential_cols.extend(list(cuts.keys()) if cuts is not None else []) # Need to get the cols being cut on

            cols = [c for c in potential_cols if c is not None]

            if sntype == "CC" and "SIM" in data_name:
                if sntypecol is None:
                    sntypecol = "TYPE"  # default sntype column name for simulated data. This is a bit hacky, but I don't want to require the user to specify this in the config file if it's always the same.
                cols.append(sntypecol)  # need the sntypecol to apply the CC cut for simulated data.
                # THIS IS EXTREMELY HACKY AND CANNOT STAY. THIS IS JUST TO SEE IF I CAN GET IT RUNNING.
            if sntype == "CC" and "DUMP" in data_name:
                if sntypecol is not None:
                    cols.append(sntypecol)  # We only need this if it is specified for splitting. This is still so hacky.
            if sntypecol is not None and "DATA" in data_name:
                cols.append(sntypecol)  # This is for fake Data only. This is really hacky now, come back and fix this.
            logging.debug(f"Columns to load for {data_name}: {cols}")

            df = self.open_dataset(path, usecols=cols)
            if "SIM" in data_name and sntype == "CC" and sntypecol != "TYPE":
                df["TYPE"] = df[sntypecol]

            # This way the later code doesn't need to worry about what the actual column names were, it can
            # just use the user submitted names.
            if submitted_subset_cols is not None:
                for col_sub, col_actual in zip(submitted_subset_cols, self.actual_subset_cols):
                    df = df.rename(columns={col_actual: col_sub})

            self.sntype = sntype
            if self.sntype not in ["IA", "CC", "all"]:
                logger.warning(f"unrecognized type: {self.sntype}")
            dataframe = pd.concat([dataframe, df])
        self.df = dataframe
        logger.debug(f"z col for {data_name}: {getattr(self, 'z_col', None)}")


    @property
    def total_counts(self):
        """Return the total number of supernovae in this dataset."""
        return len(self.df)

    @property
    def true_z_col(self):
        """ If applicable, the column name for the true simulated redshift of the SNe in this dataset."""
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
        """Calculate the counts of supernovae in redshift bins, optionally applying a classifier
        probability threshold.
        Inputs
        ------
        z_bins : array-like
            The edges of the redshift bins.
        prob_thresh : float, optional
            If provided, only include supernovae with classification probability above this threshold.
        Returns
        -------
        counts : array
            The counts of supernovae in each redshift bin.
        """
        try:
            if prob_thresh is not None:
                return binstat(self.df[self.z_col][self.prob_scone() > prob_thresh],
                            self.df[self.z_col][self.prob_scone() > prob_thresh], statistic='count', bins=z_bins)[0]

            return binstat(self.df[self.z_col], self.df[self.z_col], statistic='count', bins=z_bins)[0]
        except AttributeError:
            raise AttributeError(f"z_col is not set for {self.data_name}! Can't calculate z counts without a valid z_col. Available columns: {self.df.columns}")

    def mu_res(self):
        """Calculate the Hubble residuals for the supernovae in this dataset. Currently not used."""
        # TODO: Revisit these alpha and beta parameters
        alpha = 0.146
        beta = 3.03
        mu = 19.416 + self.df.mB + alpha * self.df.x1 - beta * self.df.c
        mu_res = mu-cosmo.distmod(self.df[self.z_col]).value
        return mu_res

    def prob_scone(self):
        """Return the classification probabilities from the SCONE classifier."""
        if self.scone_col is None:
            raise ValueError(f"No valid prob_scone column in dataset {self.data_name}!")
        return self.df[self.scone_col]

    def combine_with(self, dataset, newtype, data_name=None):
        """Combine this dataset with another SN_dataset, returning a new SN_dataset.
        Inputs
        ------
        dataset : SN_dataset
            The other dataset to combine with.
        newtype : str
            The type of the new combined dataset. Should be one of "IA", "CC", or "all".
        data_name : str, optional
            Name of the new combined dataset.
        """
        new_df = pd.concat([self.df, dataset.df], join="inner")
        if self.scone_col is not None and dataset.scone_col is not None:
            scone_prob_col = pd.concat([self.prob_scone(), dataset.prob_scone()])
            new_df["PROB_SCONE"] = scone_prob_col

        ####
        new_z_col = pd.concat([self.df[self.z_col], dataset.df[dataset.z_col]])
        new_df[self.z_col] = new_z_col

        ####

        new_dataset = SN_dataset(None, newtype, zcol=self.z_col, data_name=data_name)
        new_dataset.df = new_df
        new_dataset.z_col = self.z_col

        new_dataset.scone_col = "PROB_SCONE" if self.scone_col is not None and dataset.scone_col is not None else None
        if self.true_z_col == dataset.true_z_col:
            new_dataset.true_z_col = self.true_z_col
        else:
            raise ValueError(f"True z cols don't match for datasets being combined! {self.true_z_col} vs {dataset.true_z_col}")

        logging.debug(f"Combined dataset {data_name} has z col: {new_dataset.z_col} and scone col: {new_dataset.scone_col} and true z col: {new_dataset.true_z_col}")
        return new_dataset

        # what if the two datasets ahave different zcols? I need to determine how to do the above more properly.

        # Note that this forces the two data sets to have the
        # same z_col. I can't think of a scenario where this would be a problem, but maybe it could be.

    def apply_cut(self, col, min_val, max_val):
        """Apply a cut to the dataset on a specified column, updating the dataframe in place.
        Inputs
        ------
        col : str
            The column name to apply the cut on.
        min_val : float
            The minimum value for the cut (inclusive).
        max_val : float
            The maximum value for the cut (inclusive).
        """
        try:
            self.df = self.df[(self.df[col] >= min_val) & (self.df[col] <= max_val)]
        except KeyError:
            logging.warning("Available columns for applying cut:")
            for c in self.df.columns:
                logging.warning(f" - {c}")
            raise KeyError(f"Couldn't find column {col} in dataframe to apply cut!")

    def open_dataset(self, path, nrows=None, usecols=None):
        """Open a dataset from an unknown file type and return it as a pandas dataframe."""
        if ".FITS" in path:
            # note that since I have to fully open the fits file, this does not save memory like it does for CSVs.
            with fits.open(path) as hdul:
                fits_data = hdul[1].data
            dataframe = pd.DataFrame(np.array(fits_data))
            dataframe = dataframe[:nrows] if nrows is not None else dataframe
            if usecols is not None:
                dataframe = dataframe[usecols]

        elif ".csv" in path:
            dataframe = pd.read_csv(path, comment="#", nrows=nrows, usecols=usecols)
        else:
            dataframe = pd.read_csv(path, comment="#", sep=r"\s+", nrows=nrows, usecols=usecols)

        return dataframe

    def determine_needed_columns(self, path, zcol, subset_cols = None):
        logger.debug(f"Determining needed columns for {self.data_name} using file {path}")
        logger.debug(f"Specified zcol: {zcol}")
        # Only load the columns needed. This is important for memory management, especially for large datasets.
        if path is not None:
            test_dataframe = self.open_dataset(path, nrows=1)
            all_cols = test_dataframe.columns
        else:
            all_cols = []

        # Find the appropriate recovered redshift column.
        if zcol is not None:
            possible_z_cols = [zcol]
            logger.debug(f"zcol specified in config file: {zcol}")
            z_col_specified = True
        else:
            possible_z_cols = ['zHD', "GENZ", "HOST_ZPHOT"]
            z_col_specified = False

        self.z_col = getattr(self, "z_col", None)
        for i in possible_z_cols:
            if i in all_cols:
                if self.z_col is None:
                    self.z_col = i
                else:
                    raise ValueError(f"Multiple valid zcols found in {self.data_name}. I found: {self.z_col} and {i}")
        if self.z_col is None:
            for c in all_cols:
                logging.debug(f"Available column: {c}")
            if z_col_specified:
                raise ValueError(f"Couldn't find specified zcol {zcol} in dataframe for {self.data_name}!")
            else:
                raise ValueError(f"Couldn't find any valid zcol in dataframe for {self.data_name}!"
                                 f" I checked: {possible_z_cols}.")

        # Find the appropriate SCONE probability column, if it exists.
        scone_col = []
        for c in all_cols:
            if "PROB_SCONE" in c or "SCONE_pred" in c:
                scone_col.append(c)
        #
                # Determine or validate the dataset-level SCONE column across files.
        existing_scone_col = getattr(self, "scone_col", None)
        if existing_scone_col is None:
            # First time we are determining the SCONE column (or no SCONE column has been found yet).
            if len(scone_col) == 0:
                # No SCONE column in this file; leave self.scone_col as None for now.
                self.scone_col = None
            elif len(scone_col) > 1:
                logging.warning("TEMPORARY SCONE HACK COME BACK AND FIX THIS")
                self.scone_col = scone_col[-1]

                # raise ValueError(
                #     f"Multiple valid SCONE columns found in {self.data_name}! "
                #     f"Which do I use? I found: {scone_col}"
                # )
            else:
                # Exactly one SCONE column found; set it for the whole dataset.
                self.scone_col = scone_col[0]
        else:
            # We have already chosen a SCONE column from a previous file; enforce consistency.
            if len(scone_col) == 0:
                raise ValueError(
                    f"SCONE probability column '{existing_scone_col}' was found in an earlier file for "
                    f"{self.data_name}, but is missing in file {path}. "
                    "All files in a multi-file dataset must either all include the same SCONE column "
                    "or none at all."
                )
            elif len(scone_col) > 1:
                raise ValueError(
                    f"Multiple valid SCONE columns found in {self.data_name} for file {path}! "
                    f"Expected a single column named '{existing_scone_col}', but found: {scone_col}"
                )
            else:
                current_scone_col = scone_col[0]
                if current_scone_col != existing_scone_col:
                    raise ValueError(
                        f"Inconsistent SCONE probability column across files for {self.data_name}: "
                        f"previous files used '{existing_scone_col}', but file {path} has '{current_scone_col}'."
                    )

        if self.data_name is not None and "DATA" not in self.data_name:
            # If it's not real data, it should  have a true z column. Try to find it if it isn't specified.
            if self._true_z_col is None:
                possible_true_z_cols = ["GENZ", "TRUEZ", "SIMZ", "SIM_ZCMB"]
                cols_in_df = [col for col in possible_true_z_cols if
                              col in all_cols]
                if len(cols_in_df) > 1:
                    raise ValueError(f"Multiple possible true z cols found for {self.data_name}: {cols_in_df}. "
                                      "Please specify TRUEZCOL in config file.")
                elif len(cols_in_df) == 1:
                    self._true_z_col = cols_in_df[0]
                    logging.info(f"Auto-setting true z col for {self.data_name} to {cols_in_df[0]}")

        # Sometimes the columns used for subsetting have different names in different datasets. Like PKMJD and
        # SIM_PKMJD. This checks for close matches.
        all_good_subset_cols = []
        if subset_cols is not None:
            for scol in subset_cols:
                potential_cols = [col for col in all_cols if scol in col]
                if len(potential_cols) == 0:
                    raise ValueError(f"Couldn't find any column with {scol} in it for subsetting in {self.data_name}! Available columns: {all_cols}")
                elif len(potential_cols) > 1:
                    raise ValueError(f"Found multiple columns with {scol} in it for subsetting in {self.data_name}! Which do I use? Found: {potential_cols}")
                else:
                    all_good_subset_cols.append(potential_cols[0])
            logging.debug(f"Determined subset columns for {self.data_name}: {all_good_subset_cols}")
            return all_good_subset_cols
        else:
            return []

    def split_into_IA_and_CC(self, sntype_col, ia_vals):
        """ Split the Dataset in two, returning two new SN_datasets. Only works for datasets with a valid sntypecol."""
        if sntype_col is None:
            raise ValueError("No sntype_col specified for splitting dataset!")
        if sntype_col not in self.df.columns:
            raise ValueError(f"Couldn't find sntype_col {sntype_col} in dataframe for splitting! Available columns: {self.df.columns}")

        df_ia = self.df[self.df[sntype_col].isin(ia_vals)]
        df_cc = self.df[~self.df[sntype_col].isin(ia_vals)]

        dataset_ia = SN_dataset(None, "IA", zcol=self.z_col, data_name=f"{self.data_name.split('_ALL')[0]}_IA")
        dataset_cc = SN_dataset(None, "CC", zcol=self.z_col, data_name=f"{self.data_name.split('_ALL')[0]}_CC")

        dataset_ia.df = df_ia
        dataset_cc.df = df_cc

        dataset_ia.scone_col = self.scone_col
        dataset_cc.scone_col = self.scone_col

        dataset_ia.true_z_col = self.true_z_col
        dataset_cc.true_z_col = self.true_z_col

        dataset_ia.z_col = self.z_col
        dataset_cc.z_col = self.z_col

        return dataset_ia, dataset_cc
