import polars as pl
import numpy as np 
from plotnine import ggplot, geom_abline, stat_ecdf, geom_density, geom_ribbon, coord_cartesian, aes, labs, theme_minimal, theme, element_blank, geom_rug, geom_line, geom_hline, geom_step
from typing import Optional
import arviz as az 
from scipy import stats

def compute_pit_values(
    idata: az.InferenceData,
    var_name: str = "error",
    discrete: bool = False,
    seed: Optional[int] = None
) -> pl.DataFrame:
    """
    Compute Probability Integral Transform values for posterior predictive check.
    """
    if seed is not None:
        np.random.seed(seed)
    
    y_obs = idata.observed_data[var_name].values
    y_rep = idata.posterior_predictive[var_name].values
    n_chains, n_draws, n_obs = y_rep.shape
    y_rep = y_rep.reshape(n_chains * n_draws, n_obs)
    
    pit_values = np.zeros(n_obs)
    
    for i in range(n_obs):
        if discrete:
            p_less = np.mean(y_rep[:, i] < y_obs[i])
            p_leq = np.mean(y_rep[:, i] <= y_obs[i])
            pit_values[i] = np.random.uniform(p_less, p_leq)
        else:
            pit_values[i] = np.mean(y_rep[:, i] <= y_obs[i])
    
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
        Sample size
    alpha : float
        Significance level (0.05 = 95% envelope)
    n_points : int
        Number of points for the envelope curve
    diff : bool
        If True, center envelope at 0 (for difference plot).
        If False, center envelope at diagonal (for standard plot).
    """
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


def pp_check_pit_ecdf(
    idata: az.InferenceData,
    var_name: str = "error",
    discrete: bool = False,
    envelope: bool = True,
    alpha: float = 0.05,
    plot_diff: bool = False,
    seed: Optional[int] = None
) -> ggplot:
    """
    PIT-ECDF plot for posterior predictive checking.
    
    Parameters
    ----------
    idata : az.InferenceData
        Must contain posterior_predictive and observed_data groups
    var_name : str
        Name of the variable to check
    discrete : bool
        If True, use randomized PIT for discrete data
    envelope : bool
        Whether to show the pointwise confidence envelope
    alpha : float
        Significance level for envelope (0.05 = 95% envelope)
    plot_diff : bool
        If True, plot ECDF - uniform instead of raw ECDF
    seed : int, optional
        Random seed for randomized PIT
        
    Returns
    -------
    ggplot
    """
    pit_df = compute_pit_values(idata, var_name, discrete, seed)
    n = pit_df.height
    
    if plot_diff:
        pit_sorted = (
            pit_df
            .sort("pit")
            .with_columns(
                (pl.arange(1, n + 1) / n).alias("ecdf")
            )
            .with_columns(
                (pl.col("ecdf") - pl.col("pit")).alias("ecdf_diff")
            )
        )
        
        p = (
            ggplot(pit_sorted, aes(x="pit", y="ecdf_diff")) +
            geom_hline(yintercept=0, color="gray", linetype="dashed") +
            geom_step(color="darkblue", size=0.8) +
            labs(x="PIT", y="ECDF − Uniform") +
            theme_minimal()
        )
        
        if envelope:
            envelope_df = compute_pointwise_envelope(n, alpha, diff=True)
            p = p + geom_line(
                data=envelope_df,
                mapping=aes(x="x", y="lower"),
                color="lightblue", linetype="dashed", size=0.5
            ) + geom_line(
                data=envelope_df,
                mapping=aes(x="x", y="upper"),
                color="lightblue", linetype="dashed", size=0.5
            )
    
    else:
        p = (
            ggplot(pit_df, aes(x="pit")) +
            # geom_abline(intercept=0, slope=1, color="gray", linetype="dashed") +
            stat_ecdf(color="darkblue", size=0.8) +
            labs(x="PIT", y="ECDF") +
            coord_cartesian(xlim=(0, 1), ylim=(0, 1)) +
            theme_minimal()
        )
        
        if envelope:
            envelope_df = compute_pointwise_envelope(n, alpha, diff=False)
            p = p + geom_line(
                data=envelope_df,
                mapping=aes(x="x", y="lower"),
                color="lightblue", linetype="dashed", size=0.5
            ) + geom_line(
                data=envelope_df,
                mapping=aes(x="x", y="upper"),
                color="lightblue", linetype="dashed", size=0.5
            )
    
    return p
def pp_check_dens_overlay(
    idata: az.InferenceData, 
    var_name: str = "error", 
    num_samples: int = 100, 
    rug_on: bool = False) -> ggplot:
    """
    """
    df_ppc = az.extract(idata, group="posterior_predictive", num_samples=num_samples).to_dataframe().reset_index()
    df_ppc['.group'] = df_ppc['chain'].astype(str) + "_" + df_ppc['draw'].astype(str)
    # print(df_ppc)
    df_og = idata.observed_data.to_dataframe().reset_index()
    p = (
        ggplot() + 
        geom_density(data = df_ppc, mapping=aes(x='error', group=".group"), color="lightblue", alpha=0.5, size=0.25) + 
        geom_density(data = df_og, mapping=aes(x = 'error'), color="darkblue", size=1) + 
        theme_minimal() + 
        labs(y = "") + 
        theme(
            axis_text_y=element_blank(),  # Removes the numbers (0.0, 0.2, etc.)
            axis_ticks_y=element_blank(), # Removes the little tick marks
            panel_grid_minor_y=element_blank(), # Optional: removes minor grid lines
            panel_grid_major_y=element_blank()  # Optional: removes major grid lines
        )
    )
    if rug_on: 
        p = (p + geom_rug(data = df_og, mapping=aes(x = 'error'), alpha = 0.5))
    return p

def pp_check_ecdf_overlay(
    idata: az.InferenceData, 
    num_samples: int = 100
    ) -> ggplot: 
    """
    # TODO: some text 
    """
    df_ppc = az.extract(idata, group="posterior_predictive", num_samples=num_samples).to_dataframe().reset_index()
    df_ppc['.group'] = df_ppc['chain'].astype(str) + "_" + df_ppc['draw'].astype(str)
    df_og = idata.observed_data.to_dataframe().reset_index()
    p = (
        ggplot() + 
        stat_ecdf(data = df_ppc, mapping=aes(x='error', group=".group"), color="lightblue", alpha=0.5, size=0.25, geom="line") + 
        stat_ecdf(data = df_og, mapping=aes(x = 'error'), geom="line", color="darkblue", size=1) + 
        theme_minimal() + 
        # geom_hline(yintercept=0.5, linetype="dashed", size=0.3, color="darkblue") + 
        # geom_hline(yintercept=0, linetype="dashed", size=0.3, color="darkblue") + 
        # geom_hline(yintercept=1, linetype="dashed", size=0.3, color="darkblue") # + 
        labs(y = "") 
        # theme(
        #     axis_text_y=element_blank(),  # Removes the numbers (0.0, 0.2, etc.)
        #     axis_ticks_y=element_blank(), # Removes the little tick marks
        #     panel_grid_minor_y=element_blank(), # Optional: removes minor grid lines
        #     panel_grid_major_y=element_blank()  # Optional: removes major grid lines
        # )
    )
    return p