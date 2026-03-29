from .data import pull_cf_opportunities, load_cf_opportunities
from .model import (
    fit_all, PlayerRange,
    opportunity_weighted_range, compute_spectacular_zone,
    spectacular_play_prob, compute_all_stats,
    filter_identified,
)
from .viz import (
    plot_player_range, plot_range_comparison, plot_rankings,
    plot_opportunity_rankings, plot_spectacular_zone,
)

try:
    from .bayes import (
        BayesPlayerRange, PopulationHyperparams,
        fit_burnin, fit_season, run_sequential_pipeline,
        fit_season_evolution,
        build_sequential_priors,
        bayes_reliable_range_indicator, bayes_opportunity_weighted_range,
        bayes_compute_spectacular_zone, bayes_spectacular_play_prob,
        bayes_compute_all_stats,
        bayes_per_play_predictions,
        save_posteriors, load_posteriors,
        DEFAULT_LAM_SCHEDULE,
    )
    from .viz import (
        plot_posterior_ellipse, plot_bayes_rankings,
        plot_season_evolution, plot_prior_posterior_update,
        plot_bayes_spectacular_zone, plot_range_trajectories,
        plot_spatial_residuals, plot_calibration,
    )
except ImportError:
    pass  # cmdstanpy not installed; run: pip install -e '.[bayes]'
