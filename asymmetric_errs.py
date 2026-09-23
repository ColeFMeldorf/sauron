""" asymmetric_errors.py

Two independent ways to turn a chi-squared function into an asymmetric
1-sigma (or N-sigma) error bar on a fit parameter:

1. grid_marginalized_errors()
   Evaluate chi2 on a grid, convert to a probability density via
   L ~ exp(-chi2/2), integrate ("marginalize") out any other parameters,
   and read the 1-sigma interval off the resulting 1D PDF.
   This is a *Bayesian* answer: it implicitly assumes a flat prior over
   the grid you chose. Two different but equivalent-looking chi2 functions
   that are related by a change of variables (e.g. fitting for x vs log(x))
   will NOT in general give the same marginalized error, because "flat"
   means something different in each parameterization.

2. profile_likelihood_errors()
   For each value of the parameter of interest, *minimize* (not integrate)
   over the other parameters, then find where the resulting 1D profile
   chi2 curve rises by sigma**2 above its minimum. This is the standard
   frequentist recipe (the same one MINUIT's MINOS uses) and is the one
   that has the coverage property (~68% of the time the truth falls
   inside the 1-sigma interval) under the usual asymptotic (Wilks')
   conditions -- it does NOT depend on a prior or on how you parameterize
   the nuisance parameters.

If you have only one free parameter (no nuisance parameters to integrate
out or minimize over), both functions reduce to the same thing: just
walk chi2 along the grid and find where it crosses (min + sigma**2).

Both functions assume chi2_func(params, **kwargs) returns the *total*
-2 ln L (up to an additive constant that doesn't depend on the
parameters), i.e. the same object np.linalg.pinv/slogdet chi-squared
you'd minimize to get a best fit.
"""

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize


def _sigma_to_coverage(sigma):
    """Fraction of a 1D Gaussian's probability within +/- sigma of its mean.
    E.g. sigma=1 -> 0.6827, sigma=2 -> 0.9545. This is just so "sigma=1"
    means the usual 68.27% interval instead of you having to remember
    the number."""
    return erf(sigma / np.sqrt(2))


def grid_marginalized_errors(chi2_func, param_grids, chi2_kwargs=None,
                              sigma=1.0, vectorized=False):
    """Bayesian (flat-prior) 1-sigma interval via grid + marginalization.

    Parameters
    ----------
    chi2_func : callable
        chi2_func(params, **chi2_kwargs) -> float chi-squared value.
        `params` is a single number if there is only one entry in
        param_grids, or a 1D numpy array (one entry per parameter,
        in the same order as param_grids) otherwise.
    param_grids : list of 1D arrays
        One array per parameter, giving the grid of values to evaluate
        that parameter at. Grids do not need to be evenly spaced (we use
        the trapezoidal rule throughout, which handles that correctly),
        but they should be fine enough and wide enough that the
        likelihood has dropped to ~0 at both edges -- otherwise the
        percentiles below will be biased by truncation.
    chi2_kwargs : dict, optional
        Extra keyword arguments passed through to chi2_func every call.
    sigma : float
        How many "sigma" worth of coverage to report, e.g. 1.0 for the
        usual 68.27% interval, 2.0 for 95.45%, etc.
    vectorized : bool
        If True, and there is exactly one parameter, chi2_func is called
        ONCE with the whole grid array at once (chi2_func(grid, **kwargs))
        and must return an array of the same shape -- this is much faster
        for fine grids, but only works if chi2_func was written using
        numpy operations that broadcast over an array input rather than
        assuming a single scalar. If False (the default, and the only
        supported mode for more than one parameter), chi2_func is called
        once per grid point in an ordinary Python loop -- slower, but
        works for literally any chi2_func.

    Returns
    -------
    dict keyed by parameter index (0, 1, 2, ...), each value a dict with:
        grid, pdf, cdf   -- the 1D marginal PDF/CDF and the grid it's on
        mode             -- value of that parameter at the peak of the PDF
        median           -- 50th-percentile value
        lower_bound, upper_bound -- the sigma-coverage interval edges
        sigma_minus, sigma_plus  -- mode - lower_bound, upper_bound - mode
    Also includes 'chi2_grid' and 'likelihood_grid', the full N-D arrays,
    in case you want to sanity-check or plot the raw surface.
    """
    if chi2_kwargs is None:
        chi2_kwargs = {}
    n_params = len(param_grids)

    # np.meshgrid(*grids, indexing="ij") builds every combination of the
    # input grids at once. E.g. for two params with grids of length 5 and
    # 7, it returns two arrays of shape (5, 7): the first varies along
    # axis 0 and repeats along axis 1, the second does the opposite. This
    # is just a fast, vectorized substitute for writing nested for-loops
    # over every parameter combination.
    mesh = np.meshgrid(*param_grids, indexing="ij")

    if vectorized:
        if n_params != 1:
            raise ValueError("vectorized=True is only supported for a single parameter.")
        chi2_grid = np.asarray(chi2_func(mesh[0], **chi2_kwargs), dtype=float)
    else:
        flat_mesh = [m.ravel() for m in mesh]
        n_points = flat_mesh[0].size
        chi2_flat = np.empty(n_points)
        for i in range(n_points):
            params = flat_mesh[0][i] if n_params == 1 else np.array([m[i] for m in flat_mesh])
            chi2_flat[i] = chi2_func(params, **chi2_kwargs)
        chi2_grid = chi2_flat.reshape(mesh[0].shape)

    # L is proportional to exp(-chi2/2); subtracting the minimum first is
    # just to stop exp() from underflowing to exactly 0.0 for large chi2 --
    # it cancels out once we normalize the PDF below, so it changes nothing.
    likelihood_grid = np.exp(-0.5 * (chi2_grid - np.nanmin(chi2_grid)))

    coverage = _sigma_to_coverage(sigma)
    lower_q, upper_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2

    results = {"chi2_grid": chi2_grid, "likelihood_grid": likelihood_grid}

    # Check the edges of the chi grid. If any are >5% of the peak, warn the user that the grid may be too narrow.
    peak_val = np.nanmax(likelihood_grid)
    top_edges = [g[-1] for g in param_grids]
    bottom_edges = [g[0] for g in param_grids]
    for i, (g, top, bottom) in enumerate(zip(param_grids, top_edges, bottom_edges)):
        if likelihood_grid.take(indices=-1, axis=i).max() > 0.05 * peak_val:
            percentage = likelihood_grid.take(indices=-1, axis=i).max() / peak_val * 100
            print(f"Warning: likelihood at the top edge of parameter {i} grid ({top}) is >5% of the peak ({percentage:.2f}%).")
        if likelihood_grid.take(indices=0, axis=i).max() > 0.05 * peak_val:
            percentage = likelihood_grid.take(indices=0, axis=i).max() / peak_val * 100
            print(f"Warning: likelihood at the bottom edge of parameter {i} grid ({bottom}) is >5% of the peak ({percentage:.2f}%).")

    for i, grid_i in enumerate(param_grids):
        print(f"Processing parameter {i} with grid of length {len(grid_i)}")
        # Integrate ("marginalize") out every axis except this one, using
        # the trapezoidal rule
        marg = likelihood_grid.copy()
        for ax in sorted((a for a in range(n_params) if a != i), reverse=True):
            marg = np.trapezoid(marg, x=param_grids[ax], axis=ax)

        pdf = marg / np.trapezoid(marg, x=grid_i)

        # Cumulative trapezoidal integral -> the CDF.
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid_i))))
        cdf /= cdf[-1]  # guard against tiny numerical drift so cdf[-1] == 1 exactly

        mode = grid_i[np.argmax(pdf)]
        results[i] = {
            "grid": grid_i,
            "pdf": pdf,
            "cdf": cdf,
            "mode": mode,
            "median": np.interp(0.5, cdf, grid_i),
            "lower_bound": np.interp(lower_q, cdf, grid_i),
            "upper_bound": np.interp(upper_q, cdf, grid_i),
        }
        results[i]["sigma_minus"] = mode - results[i]["lower_bound"]
        results[i]["sigma_plus"] = results[i]["upper_bound"] - mode

    return results
