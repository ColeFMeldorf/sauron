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
    parser.add_argument("--skip-cuts", action="store_true", help="Skip applying cuts to the data.", default=False)
    parser.add_argument("--prob_thresh", type=float, default=0.5,
                        help="Probability threshold for classifying SNe as Type IA.")
    args = parser.parse_args()

    runner = sauron_runner(args)
    runner.parse_global_fit_options()

    PROB_THRESH = args.prob_thresh

    datasets, surveys = runner.unpack_dataframes()

    # Quick sanity check plot
    from matplotlib import pyplot as plt
    plt.clf()
    plt.subplot(1,2,1)
    dump_df = datasets[f"{surveys[0]}_DUMP_CC"].df
    plt.hist(dump_df['GENZ'], bins=30, alpha=0.5, label='DUMP_ALL zHD')
    sim_df = datasets[f"{surveys[0]}_SIM_CC"].df
    plt.hist(sim_df['zHD'], bins=30, alpha=0.5, label='SIM_ALL GENZ')
    #data_df = datasets[f"{surveys[0]}_DATA_CC_1"].df
    plt.yscale('log')
   # plt.hist(data_df['zHD'], bins=30, alpha=0.5, label='DATA_ALL zHD')
    plt.xlabel('Redshift')
    plt.ylabel('Counts')
    plt.subplot(1,2,2)
    dump_df = datasets[f"{surveys[0]}_DUMP_IA"].df
    plt.hist(dump_df['GENZ'], bins=30, alpha=0.5, label='DUMP_ALL zHD')
    sim_df = datasets[f"{surveys[0]}_SIM_IA"].df
    plt.hist(sim_df['zHD'], bins=30, alpha=0.5, label='SIM_ALL GENZ')
    #data_df = datasets[f"{surveys[0]}_DATA_IA_1"].df
   # plt.hist(data_df['zHD'], bins=30, alpha=0.5, label='DATA_ALL zHD')

    plt.legend()
    plt.yscale('log')
    plt.savefig('sanity_check_redshift_distribution.png')


    # Covariance calculations, if requested

    runner.calculate_covariance(PROB_THRESH=PROB_THRESH)

    for survey in surveys:
        logging.info(f"Processing survey: {survey} ========================")
        runner.get_counts(survey)  # This only gets dump counts, which have no cuts applied
        if not args.skip_cuts:
            runner.apply_cuts(survey)
        runner.calculate_transfer_matrix(survey)

        n_datasets = runner.fit_args_dict["n_datasets"][survey]
        for i in range(n_datasets):
            logging.info(f"Working on survey {survey}, dataset {i+1} -------------------")
            index = i + 1

            #plt.clf()
            #for prob_thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            runner.fit_args_dict["n_data"][survey] = \
                runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=args.debug)
            #    plt.plot(runner.fit_args_dict["n_data"][survey])
            #plt.savefig(f"varying_prob_thresh_{survey}_dataset{index}.png")

            runner.calculate_f_norm(survey, index)
            runner.fit_rate(survey)  # Should this have index?
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
