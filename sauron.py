# Working title: SAURON
# Survey-Agnostic volUmetric Rate Of superNovae

# Standard Library
import argparse
import logging

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
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Probability threshold for classifying SNe as Type IA.")
    args = parser.parse_args()

    runner = sauron_runner(args)
    runner.parse_global_fit_options()

    PROB_THRESH = args.prob_thresh

    datasets, surveys = runner.unpack_dataframes()

    # Covariance calculations, if requested

    runner.calculate_covariance(PROB_THRESH=PROB_THRESH)

    for survey in surveys:
        logging.info(f"Processing survey: {survey} ========================")
        runner.get_counts(survey)
        runner.calculate_transfer_matrix(survey)

        n_datasets = runner.fit_args_dict["n_datasets"][survey]
        for i in range(n_datasets):
            logging.info(f"Working on survey {survey}, dataset {i+1} -------------------")
            index = i + 1
            runner.calculate_f_norm(survey, index)
            runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=args.debug)
            #runner.calculate_f_norm(survey, index)
            runner.fit_rate(survey) # Should this have index?
            runner.add_results(survey, index)

    # Fit all surveys together

    if len(surveys) > 1:
        runner.fit_rate(surveys)
        runner.add_results("combined")
        surveys.extend(["combined"])


    if args.plot:
        runner.summary_plot()
    runner.save_results()


if __name__ == "__main__":
    main()
