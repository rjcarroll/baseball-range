// CF Range Model v2.0 — Sequential update version
//
// Used for all seasons after the burn-in (cf_range_v2.stan) fit.
// Per-player priors for all four parameters (a, b, y0_offset, log_gamma)
// are provided as data inputs. This encodes the carry-forward logic:
// each player's prior is their previous season's posterior, variance
// inflated by lambda to allow for year-to-year change.
//
// For new players (not seen in previous seasons), the caller supplies the
// population hyperparameter means and inflated SDs as their prior.
//
// beta_0 and beta_1 are fixed at burn-in posterior means.
// y0_offset and log_gamma are updated each season from each player's
// own data — a player who changes their starting depth will have that
// reflected in subsequent seasons.

data {
  int<lower=1> N;
  int<lower=1> P;
  array[N] int<lower=1, upper=P> player;
  vector[N] delta_x;
  vector[N] delta_y;            // feet_y - 310
  vector<lower=0>[N] hang_time;
  array[N] int<lower=0, upper=1> caught;

  // Per-player priors for a, b (from previous period, lambda-inflated)
  vector[P] prior_mu_a;
  vector<lower=0>[P] prior_sigma_a;
  vector[P] prior_mu_b;
  vector<lower=0>[P] prior_sigma_b;

  // Per-player priors for y0_offset, log_gamma (lambda-inflated)
  vector[P] prior_mu_y0_offset;
  vector<lower=0>[P] prior_sigma_y0;
  vector[P] prior_mu_log_gamma;
  vector<lower=0>[P] prior_sigma_log_gamma;

  // Logistic shape — fixed from burn-in posterior means
  real<lower=0> beta_0;
  real<lower=0> beta_1;
}

parameters {
  vector[P] a_raw;
  vector[P] b_raw;
  vector[P] y0_offset_raw;
  vector[P] log_gamma_raw;
}

transformed parameters {
  vector<lower=0>[P] a = prior_mu_a + prior_sigma_a .* a_raw;
  vector<lower=0>[P] b = prior_mu_b + prior_sigma_b .* b_raw;
  vector[P] y0_offset = prior_mu_y0_offset + prior_sigma_y0 .* y0_offset_raw;
  vector[P] log_gamma = prior_mu_log_gamma + prior_sigma_log_gamma .* log_gamma_raw;
}

model {
  a_raw         ~ std_normal();
  b_raw         ~ std_normal();
  y0_offset_raw ~ std_normal();
  log_gamma_raw ~ std_normal();

  for (n in 1:N) {
    int p = player[n];
    real dy_adj = delta_y[n] - y0_offset[p];
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
  vector[P] ellipse_area_5s;
  for (p in 1:P)
    ellipse_area_5s[p] = pi() * a[p] * b[p] * 25.0;

  vector[P] ellipse_area_5s_combined;
  for (p in 1:P)
    ellipse_area_5s_combined[p] = pi() * a[p] * b[p] * exp(log_gamma[p]) * 25.0 / 2.0
                                + pi() * a[p] * b[p] * 25.0 / 2.0;

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
