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

    # Covariance calculations, if requested

    runner.calculate_covariance(PROB_THRESH=PROB_THRESH)

    for survey in surveys:
        logging.info(f"Processing survey: {survey} ========================")
        runner.get_counts(survey)  # This only gets dump counts, which have no cuts applied
        if not args.skip_cuts:
            runner.apply_cuts(survey)

        from matplotlib import pyplot as plt
        import numpy as np
        df_fitopt = datasets["SDSS_SIM_ALL"].df
        index = 1
        df_fitopt = datasets[f"SDSS_SIM_ALL"].df
        scone_col = datasets[f"SDSS_SIM_ALL"].scone_col
        df_fitopt = df_fitopt[df_fitopt[scone_col] > PROB_THRESH]
        plt.scatter(df_fitopt["SIM_ZCMB"], df_fitopt["zPHOT"], s = 1)
        plt.plot([0,0.5], [0,0.5], color='k', linestyle='--')
        plt.xlabel("True ZCMB")
        plt.ylabel("Fitted zPHOT")
        plt.savefig(f"test_{survey.lower()}_zphot_vs_true_scone_cut.png")
        #df_fitopt.to_csv(f"test_{survey.lower()}_data_all__scone_cut.csv")

        df_passed_cuts = df_fitopt
        df_bad = df_passed_cuts[(df_passed_cuts["zPHOT"] > 0.5) & (df_passed_cuts["SIM_ZCMB"] < 0.3)]

        plt.figure(figsize=(10,10))

        plt.subplot(2,2,1)
        bins = np.linspace(-0.5, 0.5, 10)
        plt.hist(df_bad["c"] - df_bad["SIM_c"], histtype = "step", label = "Bad Photo-zs", bins = bins, density=True)
        plt.hist(df_passed_cuts["c"] - df_passed_cuts["SIM_c"], histtype = "step", label = "All Passed Cuts", bins = bins, density=True)
        plt.xlabel("c - SIM_c")
        plt.legend()

        plt.subplot(2,2,2)
        plt.hist(df_bad["x1"] - df_bad["SIM_x1"], histtype = "step", label = "Bad Photo-zs", density = True)
        plt.hist(df_passed_cuts["x1"] - df_passed_cuts["SIM_x1"], histtype = "step", label = "All Passed Cuts", density = True)
        plt.xlabel("x1 - SIM_x1")
        plt.legend()

        plt.subplot(2,2,3)
        bins = np.linspace(-20, 20, 15)
        plt.hist(df_bad["zPHOT_ERR"], histtype = "step", label = "Bad Photo-zs", bins = bins, density = True)
        plt.hist(df_passed_cuts["zPHOT_ERR"], histtype = "step", label = "All Passed Cuts", bins = bins, density = True)
        plt.xlabel("REDSHIFT_ERR")

        plt.subplot(2,2,4)
        bins = np.linspace(0, 1, 15)
        plt.hist(df_bad["FITPROB"], histtype = "step", label = "Bad Photo-zs", density = True, bins = bins)
        plt.hist(df_passed_cuts["FITPROB"], histtype = "step", label = "All Passed Cuts", density = True, bins = bins)
        plt.xlabel("FITPROB")

        plt.savefig(f"test_{survey.lower()}_bad_photoz_diagnostics.png")
        runner.calculate_transfer_matrix(survey)

        n_datasets = runner.fit_args_dict["n_datasets"][survey]
        for i in range(n_datasets):
            logging.info(f"Working on survey {survey}, dataset {i+1} -------------------")
            index = i + 1

            runner.fit_args_dict["n_data"][survey] = \
                runner.calculate_CC_contamination(PROB_THRESH, index, survey, debug=args.debug)
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
