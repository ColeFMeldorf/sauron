# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library

import argparse
import pandas as pd
import numpy as np

# Astronomy
from astropy.cosmology import LambdaCDM

# Sauron modules
from runner import sauron_runner
from funcs import calculate_covariance_matrix_term, rescale_CC_for_cov

cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)

# These need to be added to config file later


def main():
    parser = argparse.ArgumentParser(description='SAURON: Survey-Agnostic volUmetric Rate Of superNovae')
    parser.add_argument('config', help='Path to the config file (positional argument)')
    parser.add_argument('--output', '-o', default='sauron_output.csv', help='Path to the output file (optional)')
    parser.add_argument("--cheat_cc", action="store_true", help="Cheat and skip CC step. Data_IA will be used as"
                        " Data_All.")
    parser.add_argument("-c", "--covariance", action="store_true", help="Calculate covariance matrix terms.", default=True)
    parser.add_argument("-p", "--plot", action="store_true", help="Generate diagnostic plots.", default=False)
    args = parser.parse_args()

    runner = sauron_runner(args)
    runner.parse_global_fit_options()

    datasets, surveys = runner.unpack_dataframes()
    # Covariance calculations, if requested
    PROB_THRESH = 0.13
    runner.calculate_covariance(PROB_THRESH=PROB_THRESH)

    for survey in surveys:
        print(f"Processing survey: {survey} ========================")
        runner.get_counts(survey)
        runner.results[survey] = []
        runner.final_counts[survey] = {}
        runner.calculate_transfer_matrix(survey)

        n_datasets = runner.fit_args_dict["n_datasets"][survey]
        for i in range(n_datasets):
            print(f"Working on survey {survey}, dataset {i+1} -------------------")
            # Core Collapse Contamination
            index = i + 1
            runner.fit_args_dict["n_data"][survey] = \
                runner.calculate_CC_contamination(PROB_THRESH, index, survey)
            n_data = runner.fit_args_dict["n_data"][survey]

            # This can't stay actually, we can't used DATA_IA because we won't have it irl.

            z_bins = runner.fit_args_dict['z_bins'][survey]

            if runner.fit_args_dict["cc_are_sep"][survey]:
                f_norm = np.sum(datasets[f"{survey}_DATA_IA_{index}"].z_counts(z_bins)) / \
                    np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))

            else:
                f_norm = np.sum(datasets[f"{survey}_DATA_ALL_{index}"].z_counts(z_bins)) / \
                    np.sum(datasets[f"{survey}_SIM_ALL"].z_counts(z_bins))

            runner.fit_args_dict['f_norm'][survey] = f_norm

            # f_norm = np.sum(n_data) / \
            #     np.sum(datasets[f"{survey}_SIM_IA"].z_counts(z_bins))
            print(f"Calculated f_norm to be {f_norm}")

            result, chi, Ei, cov, Ei_err = runner.fit_rate(survey)

            runner.final_counts[survey]["predicted_counts"] = Ei
            runner.final_counts[survey]["observed_counts"] = n_data
            runner.final_counts[survey]["predicted_counts_err"] = Ei_err
            runner.n_data = n_data
            z_bins = runner.fit_args_dict['z_bins'][survey]

            # This needs to be updated for more parameters later
            if n_datasets > 1:
                survey_name = survey + f"_dataset_{i+1}"
            else:
                survey_name = survey

            runner.results[survey].append(pd.DataFrame({
                "delta_alpha": result[0],
                "delta_beta": result[1],
                "reduced_chi_squared": chi/(len(z_bins)-2),
                "alpha_error": np.sqrt(cov[0, 0]),
                "beta_error": np.sqrt(cov[1, 1]),
                "cov_alpha_beta": cov[0, 1],
                "survey": survey_name
                }, index=np.array([0])))

    # Fit all surveys together

    if len(surveys) > 1:
        result, chi, Ei, cov, Ei_err = runner.fit_rate(surveys)
        runner.results["combined"] = pd.DataFrame({
                "delta_alpha": result[0],
                "delta_beta": result[1],
                "reduced_chi_squared": chi/(len(z_bins)-2),
                "alpha_error": np.sqrt(cov[0, 0]),
                "beta_error": np.sqrt(cov[1, 1]),
                "cov_alpha_beta": cov[0, 1],
                "survey": "combined"
                }, index=np.array([0]))
        surveys.extend(["combined"])

    if args.plot:
        runner.summary_plot()
    runner.save_results()


if __name__ == "__main__":
    main()
