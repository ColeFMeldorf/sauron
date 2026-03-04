# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library
import argparse
import logging
import math

# Sauron modules
from runner import sauron_runner

# Configure the basic logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    parser = argparse.ArgumentParser(description='SAURON: Survey-Agnostic volUmetric Rate Of superNovae')
    parser.add_argument('config', help='Path to the config file (positional argument)')
    parser.add_argument('--output', '-o', default='sauron_output.csv', help='Path to the output file (optional)')
    parser.add_argument("--cheat_cc", action="store_true", help="Cheat and skip CC step. Data_IA will be used as"
                        " Data_All.")
    parser.add_argument("--sys_cov", "--systematic_covariance", action=argparse.BooleanOptionalAction,
                        help="Calculate systematic covariance matrix terms.", default=True)
    parser.add_argument("-p", "--plot", action="store_true", help="Generate diagnostic plots.", default=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.", default=False)
    parser.add_argument("--skip-cuts", action="store_true", help="Skip applying cuts to the data.", default=False)
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Probability threshold for classifying SNe as Type IA.")
    parser.add_argument("--sanity-check", action=argparse.BooleanOptionalAction, help="Perform sanity checks"
                        " that simulations look reasonable.", default=True)
    parser.add_argument("--fit-only-one-combined", "--fit1", action=argparse.BooleanOptionalAction, help="Only fit one combined dataset across all"
                        " surveys, instead of fitting as many as there are datasets. I.e., if I have 5"
                        "simulated datasets and 10 for another, I could do 5 combined datasets if this is set to False.", default=True)
    args = parser.parse_args()

    runner = sauron_runner(args)
    runner.parse_global_fit_options()

    PROB_THRESH = args.prob_thresh

    datasets, surveys = runner.unpack_dataframes()

    # Covariance calculations, if requested

    runner.calculate_covariance(PROB_THRESH=PROB_THRESH)

    for survey in surveys:
        logging.info(f"Processing survey: {survey} ========================")
        runner.apply_cuts(survey, subset_version=True) # This takes a subset of the data if asked for.
        runner.get_counts(survey)  # This only gets dump counts, which have no cuts applied
        if args.sanity_check:
            runner.perform_sanity_checks(survey)
        if not args.skip_cuts:
            runner.apply_cuts(survey)
        runner.calculate_transfer_matrix(survey)

        n_datasets = runner.fit_args_dict["n_datasets"][survey]
        runner.load_and_decontaminate_datasets(survey, PROB_THRESH=PROB_THRESH)

        for i in range(n_datasets):
            logging.info(f"Working on survey {survey}, dataset {i+1} -------------------")
            index = i + 1
            runner.calculate_f_norm(survey, index)
            runner.fit_rate(survey, index)
            runner.add_results(survey, index)

    # Fit all surveys together


    if len(surveys) > 1:
        logging.debug("Starting combo fit with surveys: " + ", ".join(surveys))
        runner.load_and_decontaminate_datasets(surveys, PROB_THRESH=PROB_THRESH)  # index -1 is for combined datasets
        if args.fit_only_one_combined:
            logging.info("Fitting only one combined dataset across all surveys.")
            indices = [0]
        else:
            total_possible_indexes = math.prod([runner.fit_args_dict["n_datasets"][s] for s in surveys])
            indices = range(1, total_possible_indexes + 1)

        runner.fit_args_dict["n_datasets"]["combined"] = len(indices)  # Update the number of datasets for the combined survey to reflect the number of combinations of datasets across surveys.

        for index in indices:
            logging.info(f"Fitting all possible combined dataset across all surveys, index {index} -----------------------")
            runner.fit_rate(surveys, index=index)
            runner.add_results("combined", index=index)
        surveys.extend(["combined"])

    if args.plot:
        runner.summary_plot()
    runner.save_results()


if __name__ == "__main__":
    main()
