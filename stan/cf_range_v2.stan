// CF Range Model v2.0 — Bayesian hierarchical burn-in
//
// Extends v1.0 with two new per-player parameters:
//
//   y0_offset[p] — starting depth offset from canonical 310 ft.
//                  True starting depth = 310 + y0_offset.
//                  Adjusted displacement: dy_adj = delta_y - y0_offset[p]
//                  (delta_y is already feet_y - 310 in the data).
//
//   log_gamma[p] — log of charge/retreat asymmetry factor γ.
//                  b_eff = b * exp(log_gamma) when ball is in front of fielder
//                         (dy_adj < 0, charging)
//                  b_eff = b                  when ball is behind fielder
//                         (dy_adj > 0, retreating)
//
// Motivation: v1.0 diagnostics showed mean residual -0.06 in the charging zone
// (225-315 degrees), consistent with fielders playing deeper than 310 ft.
// y0_offset absorbs this positional confound; log_gamma then captures genuine
// directional asymmetry in running speed.
//
// Smooth blending: the step between b and b*gamma is replaced by a sigmoid
// transition (width 5 ft) for HMC differentiability. At |dy_adj| > 15 ft
// the blend is >95% saturated, so the approximation is accurate for most plays.

data {
  int<lower=1> N;
  int<lower=1> P;
  array[N] int<lower=1, upper=P> player;
  vector[N] delta_x;
  vector[N] delta_y;            // feet_y - 310 (canonical CF starting depth)
  vector<lower=0>[N] hang_time;
  array[N] int<lower=0, upper=1> caught;
}

parameters {
  // v1.0 hyperparameters
  real<lower=1> mu_a;
  real<lower=1> mu_b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;

  // v2.0 hyperparameters
  real mu_y0_offset;              // population mean depth offset (ft from 310)
  real<lower=0> sigma_y0;         // population SD of depth offsets
  real mu_log_gamma;              // population mean log(gamma)
  real<lower=0> sigma_log_gamma;  // population SD of log(gamma)

  // Per-player parameters (non-centered for HMC efficiency)
  vector[P] a_raw;
  vector[P] b_raw;
  vector[P] y0_offset_raw;
  vector[P] log_gamma_raw;

  // Logistic shape (shared across players)
  real<lower=0> beta_0;
  real<lower=0> beta_1;
}

transformed parameters {
  vector<lower=0>[P] a = mu_a + sigma_a * a_raw;
  vector<lower=0>[P] b = mu_b + sigma_b * b_raw;
  vector[P] y0_offset = mu_y0_offset + sigma_y0 * y0_offset_raw;
  vector[P] log_gamma = mu_log_gamma + sigma_log_gamma * log_gamma_raw;
}

model {
  // v1.0 hyperpriors
  mu_a    ~ normal(15, 5);
  mu_b    ~ normal(18, 6);
  sigma_a ~ exponential(0.5);
  sigma_b ~ exponential(0.5);

  // v2.0 hyperpriors
  // y0_offset: expect fielders to be near 310 ft, allow up to ~20 ft spread
  mu_y0_offset    ~ normal(0, 10);
  sigma_y0        ~ normal(0, 10);     // half-normal (constrained positive)
  // gamma: symmetric prior; data determines direction
  mu_log_gamma    ~ normal(0, 0.3);
  sigma_log_gamma ~ normal(0, 0.3);

  // Non-centered player parameters
  a_raw         ~ std_normal();
  b_raw         ~ std_normal();
  y0_offset_raw ~ std_normal();
  log_gamma_raw ~ std_normal();

  // Logistic shape
  beta_0 ~ normal(4, 2);
  beta_1 ~ normal(4, 2);

  // Likelihood
  for (n in 1:N) {
    int p = player[n];
    real dy_adj = delta_y[n] - y0_offset[p];
    // Smooth blend: charging weight = inv_logit(-dy_adj / 5)
    // = 1 when dy_adj << 0 (clearly charging), 0 when dy_adj >> 0 (clearly retreating)
    real charge_weight = inv_logit(-dy_adj / 5.0);
    real b_eff = b[p] * exp(log_gamma[p] * charge_weight);
    real d = sqrt(
      square(delta_x[n] / (a[p] * hang_time[n])) +
      square(dy_adj / (b_eff * hang_time[n]))
    );
    caught[n] ~ bernoulli_logit(beta_0 - beta_1 * d);
  }
}

generated quantities {
  // Retreating ellipse area at tau = 5s (b, not b*gamma)
  vector[P] ellipse_area_5s;
  for (p in 1:P)
    ellipse_area_5s[p] = pi() * a[p] * b[p] * 25.0;

  // Combined area: average of charging (b*gamma) and retreating (b) halves
  vector[P] ellipse_area_5s_combined;
  for (p in 1:P)
    ellipse_area_5s_combined[p] = pi() * a[p] * b[p] * exp(log_gamma[p]) * 25.0 / 2.0
                                + pi() * a[p] * b[p] * 25.0 / 2.0;

  // Log-likelihood per observation (for LOO-CV)
  vector[N] log_lik;
  for (n in 1:N) {
    int p = player[n];
    real dy_adj = delta_y[n] - y0_offset[p];
    real charge_weight = inv_logit(-dy_adj / 5.0);
    real b_eff = b[p] * exp(log_gamma[p] * charge_weight);
    real d = sqrt(
      square(delta_x[n] / (a[p] * hang_time[n])) +
      square(dy_adj / (b_eff * hang_time[n]))
    );
    log_lik[n] = bernoulli_logit_lpmf(caught[n] | beta_0 - beta_1 * d);
  }
}
