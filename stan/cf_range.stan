// CF Range Model — Bayesian hierarchical version
//
// Model:
//   d[n]          = sqrt( (dx[n] / (a[p] * tau[n]))^2 + (dy[n] / (b[p] * tau[n]))^2 )
//   caught[n]     ~ Bernoulli( sigma(beta_0 - beta_1 * d[n]) )
//   a[p]          ~ Normal+(mu_a, sigma_a)   [partial pooling]
//   b[p]          ~ Normal+(mu_b, sigma_b)
//
// (a[p], b[p]): lateral and depth speed parameters for player p (ft/sec)
// (beta_0, beta_1): shared logistic shape; beta_0 ≈ beta_1 places the 50%
//                   boundary at d = 1 (the ellipse edge)
//
// Partial pooling: players are shrunk toward population means (mu_a, mu_b).
// This is valuable for players with few opportunities.
//
// R_i as a random set: because a[p] and b[p] have full posteriors, the
// ellipse boundary is itself a random set. The credible band around the
// boundary is uncertainty about which set it is—not just a scalar interval.

data {
  int<lower=1> N;                   // total ball-player-outcome observations
  int<lower=1> P;                   // number of players
  array[N] int<lower=1, upper=P> player;  // player index for each observation
  vector[N] delta_x;                // lateral displacement (ft)
  vector[N] delta_y;                // depth displacement (ft)
  vector<lower=0>[N] hang_time;     // hang time (seconds)
  array[N] int<lower=0, upper=1> caught;  // outcome
}

parameters {
  // Population hyperparameters
  real<lower=1> mu_a;               // population mean lateral speed (ft/sec)
  real<lower=1> mu_b;               // population mean depth speed (ft/sec)
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;

  // Per-player speeds (non-centered parameterization for HMC efficiency)
  vector[P] a_raw;
  vector[P] b_raw;

  // Logistic shape (shared across players)
  real<lower=0> beta_0;
  real<lower=0> beta_1;
}

transformed parameters {
  vector<lower=0>[P] a = mu_a + sigma_a * a_raw;
  vector<lower=0>[P] b = mu_b + sigma_b * b_raw;
}

model {
  // Hyperpriors — centered on typical CF range speeds
  mu_a    ~ normal(15, 5);
  mu_b    ~ normal(18, 6);
  sigma_a ~ exponential(0.5);
  sigma_b ~ exponential(0.5);

  // Non-centered player parameters
  a_raw ~ std_normal();
  b_raw ~ std_normal();

  // Logistic shape: beta_0 = beta_1 puts the 50% boundary at d=1
  beta_0 ~ normal(4, 2);
  beta_1 ~ normal(4, 2);

  // Likelihood
  for (n in 1:N) {
    int p = player[n];
    real d = sqrt(
      square(delta_x[n] / (a[p] * hang_time[n])) +
      square(delta_y[n] / (b[p] * hang_time[n]))
    );
    caught[n] ~ bernoulli_logit(beta_0 - beta_1 * d);
  }
}

generated quantities {
  // Ellipse area at tau_h = 5 seconds (summary statistic for ranking)
  // area_i = pi * (a_i * 5)^2 * (b_i / a_i) = pi * a_i * b_i * 25
  vector[P] ellipse_area_5s;
  for (p in 1:P)
    ellipse_area_5s[p] = pi() * a[p] * b[p] * 25.0;

  // Posterior predictive check: log-lik per observation (for LOO-CV)
  vector[N] log_lik;
  for (n in 1:N) {
    int p = player[n];
    real d = sqrt(
      square(delta_x[n] / (a[p] * hang_time[n])) +
      square(delta_y[n] / (b[p] * hang_time[n]))
    );
    log_lik[n] = bernoulli_logit_lpmf(caught[n] | beta_0 - beta_1 * d);
  }
}
