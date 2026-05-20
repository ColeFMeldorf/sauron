

def debug_covariance_matrix_plot(reduced_cov, reduced_cov_thresh, reduced_cov_rate_norm, survey):
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.imshow(reduced_cov, origin="lower")
    plt.colorbar(label="Reduced Covariance")
    plt.title(f"Reduced Systematic Covariance Matrix for {survey}")
    plt.xlabel("Redshift Bin")
    plt.ylabel("Redshift Bin")
    plt.subplot(1, 3, 2)
    plt.imshow(reduced_cov_thresh, origin="lower")
    plt.colorbar(label="Reduced Covariance")
    plt.title(f"Reduced Threshold Covariance Matrix for {survey}")
    plt.xlabel("Redshift Bin")
    plt.ylabel("Redshift Bin")
    plt.subplot(1, 3, 3)
    plt.imshow(reduced_cov_rate_norm, origin="lower")
    plt.colorbar(label="Reduced Covariance")
    plt.title(f"Reduced Rate Norm Covariance Matrix for {survey}")
    plt.xlabel("Redshift Bin")
    plt.ylabel("Redshift Bin")
    path = f"plots/cov_sys_{survey}.png"
    logging.debug(f"Saving systematic covariance plot to {path}")
    plt.savefig(path)

def efficiency_matrix_debug_plot(eff_ij, survey, z_bins):
    LaurenNicePlots()
    plt.clf()
    plt.figure(figsize=(7, 6), dpi = 200)
    plt.imshow(eff_ij[1:-1, :].T, origin="lower", aspect="auto",
                extent=[z_bins[0], z_bins[-1], z_bins[0], z_bins[-1]],
                vmin=0, vmax=np.max(eff_ij)
                )
    plt.colorbar(label=r"Efficiency (n$_{\mathrm{obs}}$ / n$_{\mathrm{sim}}$)")
    plt.title(f"Efficiency Matrix for {survey}")
    plt.ylabel("Recovered Redshift")
    plt.xlabel("True Redshift")
    path = f"plots/efficiency_matrix_{survey}.png"
    logging.debug(f"Saving efficiency matrix plot to {path}")
    plt.savefig(path)