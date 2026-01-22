import polars as pl
import numpy as np 
from plotnine import (
    ggplot, geom_abline, stat_ecdf, geom_density, geom_ribbon, 
    coord_cartesian, aes, labs, theme_minimal, theme, element_blank, 
    geom_rug, geom_line, geom_hline, geom_step
)
from typing import Optional
import arviz as az 
from scipy import stats
from scipy.optimize import minimize_scalar


def _validate_idata(idata: az.InferenceData, var_name: str, needs_posterior_predictive: bool = True) -> None:
    """Validate InferenceData object has required groups and variables."""
    if not isinstance(idata, az.InferenceData):
        raise TypeError(f"idata must be az.InferenceData, got {type(idata)}")
    
    if not hasattr(idata, "observed_data"):
        raise ValueError("idata must contain 'observed_data' group.")
    
    if var_name not in idata.observed_data:
        available = list(idata.observed_data.keys())
        raise ValueError(f"Variable '{var_name}' not found in observed_data. Available: {available}")
    
    if needs_posterior_predictive:
        if not hasattr(idata, "posterior_predictive"):
            raise ValueError(
                "idata must contain 'posterior_predictive' group. "
                "Run pm.sample_posterior_predictive first."
            )
        if var_name not in idata.posterior_predictive:
            available = list(idata.posterior_predictive.keys())
            raise ValueError(f"Variable '{var_name}' not found in posterior_predictive. Available: {available}")

def _p_interior(p_int: np.ndarray, x1: np.ndarray, x2: np.ndarray,
                z1: float, z2: float, N: int) -> np.ndarray:
    """
    Probability that a scaled ECDF stays within bounds between two evaluation points.
    
    Parameters
    ----------
    p_int : np.ndarray
        For each value in x1, the probability that ECDF stayed within bounds
        until z1 and takes that value at z1
    x1 : np.ndarray
        Scaled ECDF values at left endpoint z1
    x2 : np.ndarray
        Scaled ECDF values at right endpoint z2
    z1 : float
        Left evaluation point in [0, 1]
    z2 : float
        Right evaluation point in [0, 1], z2 > z1
    N : int
        Total sample size
    
    Returns
    -------
    np.ndarray
        Probability of transitioning from x1 to each value in x2
    """
    if z1 >= 1:
        z_tilde = 0.0
    else:
        z_tilde = (z2 - z1) / (1 - z1)
    
    x_diff = np.subtract.outer(x2, x1)  # shape: (len(x2), len(x1))
    N_tilde = N - x1  # shape: (len(x1),)
    
    p_x2_int = np.zeros((len(x2), len(x1)))
    for j, (n_t, p_i) in enumerate(zip(N_tilde, p_int)):
        if n_t > 0:
            p_x2_int[:, j] = p_i * stats.binom.pmf(x_diff[:, j], int(n_t), z_tilde)
        else:
            p_x2_int[:, j] = p_i * (x_diff[:, j] == 0).astype(float)
    
    return p_x2_int.sum(axis=1)

def _adjust_gamma_optimize(N: int, K: int, prob: float) -> float:
    """
    Find gamma such that simultaneous confidence bands achieve desired coverage.
    
    Parameters
    ----------
    N : int
        Sample size
    K : int
        Number of evaluation points
    prob : float
        Desired simultaneous coverage (e.g., 0.99)
    
    Returns
    -------
    float
        Adjusted gamma parameter
    """
    def target(gamma: float) -> float:
        z = np.arange(1, K) / K
        z1 = np.concatenate([[0], z])
        z2 = np.concatenate([z, [1]])
        
        x2_lower = stats.binom.ppf(gamma / 2, N, z2).astype(int)
        x2_upper = np.concatenate([N - x2_lower[1:][::-1], [N]]).astype(int)
        
        x1 = np.array([0])
        p_int = np.array([1.0])
        
        for i in range(len(z1)):
            x2_range = np.arange(x2_lower[i], x2_upper[i] + 1)
            p_int = _p_interior(p_int, x1, x2_range, z1[i], z2[i], N)
            x1 = x2_range
        
        return abs(prob - p_int.sum())
    
    result = minimize_scalar(target, bounds=(0, 1 - prob), method='bounded')
    return result.x

def compute_pit_values(
    idata: az.InferenceData,
    var_name: str = "error",
    seed: Optional[int] = None
) -> pl.DataFrame:
    """
    Compute Probability Integral Transform values following bayesplot's logic.
    
    Uses randomized PIT: p(yrep < y) + U(0, p(yrep == y))
    This handles both continuous and discrete data correctly.
    
    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData with posterior_predictive and observed_data
    var_name : str, default="error"
        Variable name in both groups
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    pl.DataFrame
        Columns: obs_idx, y_obs, pit
    """
    if seed is not None:
        np.random.seed(seed)
    
    y_obs = idata.observed_data[var_name].values
    y_rep = idata.posterior_predictive[var_name].values
    
    # Flatten chains and draws: (chains, draws, obs) -> (samples, obs)
    n_chains, n_draws, n_obs = y_rep.shape
    y_rep = y_rep.reshape(n_chains * n_draws, n_obs)
    
    pit_values = np.zeros(n_obs)
    
    for i in range(n_obs):
        p_less = np.mean(y_rep[:, i] < y_obs[i])
        p_equal = np.mean(y_rep[:, i] == y_obs[i])
        pit_values[i] = p_less + np.random.uniform(0, p_equal)
    
    return pl.DataFrame({
        "obs_idx": np.arange(n_obs),
        "y_obs": y_obs,
        "pit": pit_values
    })

def compute_pointwise_envelope(
    n: int,
    alpha: float = 0.05,
    n_points: int = 100,
    diff: bool = False
) -> pl.DataFrame:
    """
    Compute pointwise confidence envelope for uniform ECDF.
    
    Parameters
    ----------
    n : int
        Sample size (number of observations).
    alpha : float, default=0.05
        Significance level (0.05 = 95% envelope).
    n_points : int, default=100
        Number of points for the envelope curve.
    diff : bool, default=False
        If True, center envelope at 0 (for difference plot).
        If False, center envelope at diagonal (for standard plot).
        
    Returns
    -------
    pl.DataFrame
        DataFrame with columns: x, lower, upper
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    x = np.linspace(0, 1, n_points)
    se = np.sqrt(x * (1 - x) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    
    if diff:
        lower = -z * se
        upper = z * se
    else:
        lower = np.clip(x - z * se, 0, 1)
        upper = np.clip(x + z * se, 0, 1)
    
    return pl.DataFrame({
        "x": x,
        "lower": lower,
        "upper": upper
    })

def compute_simultaneous_envelope(
    N: int,
    prob: float = 0.99,
    K: Optional[int] = None
) -> pl.DataFrame:
    """
    Compute simultaneous confidence envelope for uniform ECDF.
    """
    if K is None:
        K = min(N + 1, 1000)
    
    gamma = _adjust_gamma_optimize(N=N, K=K, prob=prob)
    
    x = np.linspace(0, 1, K + 1)
    lower = stats.binom.ppf(gamma / 2, N, x) / N
    upper = stats.binom.ppf(1 - gamma / 2, N, x) / N
    
    # Drop the first element (x=0) to match bayesplot behavior
    return pl.DataFrame({
        "x": x[1:],
        "lower": lower[1:],
        "upper": upper[1:],
        "lower_diff": lower[1:] - x[1:],
        "upper_diff": upper[1:] - x[1:]
    })

def pp_check_pit_ecdf(
    idata: az.InferenceData,
    var_name: str = "error",
    envelope: bool = True,
    prob: float = 0.99,
    K: Optional[int] = None,
    plot_diff: bool = False,
    seed: Optional[int] = None
) -> ggplot:
    """
    PIT-ECDF plot for posterior predictive checking.
    
    Parameters
    ----------
    idata : az.InferenceData
        Must contain posterior_predictive and observed_data groups.
    var_name : str, default="error"
        Name of the variable to check.
    envelope : bool, default=True
        Whether to show the simultaneous confidence envelope.
    prob : float, default=0.99
        Simultaneous coverage probability for envelope.
    K : int, optional
        Number of evaluation points for envelope. Defaults to min(N + 1, 1000).
    plot_diff : bool, default=False
        If True, plot ECDF - uniform instead of raw ECDF.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    ggplot
        A plotnine ggplot object.
        
    Examples
    --------
    >>> pp_check_pit_ecdf(idata)
    >>> pp_check_pit_ecdf(idata, plot_diff=True)
    """
    pit_df = compute_pit_values(idata, var_name, seed=seed)
    N = pit_df.height
    
    if K is None:
        K = min(N + 1, 1000)
    
    if plot_diff:
        pit_sorted = (
            pit_df
            .sort("pit")
            .with_columns(
                (pl.arange(1, N + 1) / N).alias("ecdf")
            )
            .with_columns(
                (pl.col("ecdf") - pl.col("pit")).alias("ecdf_diff")
            )
        )
        
        p = (
            ggplot(pit_sorted, aes(x="pit", y="ecdf_diff")) +
            geom_step(color="darkblue", size=0.8) +
            geom_hline(yintercept=0, linetype="dotted", alpha=0.5) +
            labs(x="PIT", y="ECDF − Uniform")
        )
        
        if envelope:
            envelope_df = compute_simultaneous_envelope(N=N, prob=prob, K=K)
            p = (
                p + 
                geom_step(
                    data=envelope_df, 
                    mapping=aes(x="x", y="lower_diff"),
                    color="lightblue", linetype="dashed", size=0.5
                ) + 
                geom_step(
                    data=envelope_df, 
                    mapping=aes(x="x", y="upper_diff"),
                    color="lightblue", linetype="dashed", size=0.5
                )
            )
    
    else:
        p = (
            ggplot(pit_df, aes(x="pit")) +
            stat_ecdf(color="darkblue", size=0.8) +
            geom_abline(intercept=0, slope=1, linetype="dotted", alpha=0.5) +
            labs(x="PIT", y="ECDF") +
            coord_cartesian(xlim=(0, 1), ylim=(0, 1))
        )
        
        if envelope:
            envelope_df = compute_simultaneous_envelope(N=N, prob=prob, K=K)
            p = (
                p + 
                geom_step(
                    data=envelope_df, 
                    mapping=aes(x="x", y="lower"),
                    color="lightblue", linetype="dashed", size=0.5
                ) + 
                geom_step(
                    data=envelope_df, 
                    mapping=aes(x="x", y="upper"),
                    color="lightblue", linetype="dashed", size=0.5
                )
            )
    
    return p + theme_minimal()

def pp_check_dens_overlay(
    idata: az.InferenceData, 
    var_name: str = "error", 
    num_samples: int = 100, 
    rug_on: bool = False
) -> ggplot:
    """
    Density overlay posterior predictive check.
    
    Overlays density plots of posterior predictive samples (light blue)
    on the observed data density (dark blue).
    
    Parameters
    ----------
    idata : az.InferenceData
        Must contain posterior_predictive and observed_data groups.
    var_name : str, default="error"
        Name of the variable to check.
    num_samples : int, default=100
        Number of posterior predictive samples to overlay.
    rug_on : bool, default=False
        If True, add a rug plot of observed values.
        
    Returns
    -------
    ggplot
        A plotnine ggplot object.
    """
    _validate_idata(idata, var_name, needs_posterior_predictive=True)
    
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    
    df_ppc = az.extract(idata, group="posterior_predictive", num_samples=num_samples).to_dataframe().reset_index()
    df_ppc['.group'] = df_ppc['chain'].astype(str) + "_" + df_ppc['draw'].astype(str)
    df_og = idata.observed_data.to_dataframe().reset_index()
    
    p = (
        ggplot() + 
        geom_density(data=df_ppc, mapping=aes(x=var_name, group=".group"), 
                     color="lightblue", alpha=0.5, size=0.25) + 
        geom_density(data=df_og, mapping=aes(x=var_name), color="darkblue", size=1) + 
        labs(y="") + 
        theme_minimal() + 
        theme(
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            panel_grid_minor_y=element_blank(),
            panel_grid_major_y=element_blank()
        )
    )
    
    if rug_on: 
        p = p + geom_rug(data=df_og, mapping=aes(x=var_name), alpha=0.5)
    
    return p


def pp_check_ecdf_overlay(
    idata: az.InferenceData, 
    var_name: str = "error",
    num_samples: int = 100
) -> ggplot: 
    """
    ECDF overlay posterior predictive check.
    
    Overlays empirical CDFs of posterior predictive samples (light blue)
    on the observed data ECDF (dark blue).
    
    Parameters
    ----------
    idata : az.InferenceData
        Must contain posterior_predictive and observed_data groups.
    var_name : str, default="error"
        Name of the variable to check.
    num_samples : int, default=100
        Number of posterior predictive samples to overlay.
        
    Returns
    -------
    ggplot
        A plotnine ggplot object.
    """
    _validate_idata(idata, var_name, needs_posterior_predictive=True)
    
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    
    df_ppc = az.extract(idata, group="posterior_predictive", num_samples=num_samples).to_dataframe().reset_index()
    df_ppc['.group'] = df_ppc['chain'].astype(str) + "_" + df_ppc['draw'].astype(str)
    df_og = idata.observed_data.to_dataframe().reset_index()
    
    p = (
        ggplot() + 
        stat_ecdf(data=df_ppc, mapping=aes(x=var_name, group=".group"), 
                  color="lightblue", alpha=0.5, size=0.25, geom="line") + 
        stat_ecdf(data=df_og, mapping=aes(x=var_name), geom="line", 
                  color="darkblue", size=1) + 
        labs(y="") +
        theme_minimal()
    )
    
    return p