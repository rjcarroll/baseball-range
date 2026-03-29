// CF Range Model — Sequential update version
//
// Used for all seasons after the burn-in (cf_range.stan) fit.
// Per-player prior parameters are provided as data inputs rather than
// estimated from the population. This encodes the carry-forward logic:
// each player's prior is their previous season's posterior, inflated by
// a factor lambda to allow for year-to-year change.
//
// For new players (not seen in the burn-in or previous seasons), the caller
// supplies the population hyperparameter means and inflated SDs as their prior.
//
// beta_0 and beta_1 (logistic shape) are fixed at the burn-in posterior means.
// Re-estimating them from a single season would be poorly identified.

data {
  int<lower=1> N;                          // total observations
  int<lower=1> P;                          // number of players this season
  array[N] int<lower=1, upper=P> player;  // player index for each observation
  vector[N] delta_x;                       // lateral displacement (ft)
  vector[N] delta_y;                       // depth displacement (ft)
  vector<lower=0>[N] hang_time;            // hang time (seconds)
  array[N] int<lower=0, upper=1> caught;  // outcome

  // Per-player prior parameters (from previous period, lambda-inflated)
  vector[P] prior_mu_a;
  vector<lower=0>[P] prior_sigma_a;
  vector[P] prior_mu_b;
  vector<lower=0>[P] prior_sigma_b;

  // Logistic shape — fixed from burn-in posterior means
  real<lower=0> beta_0;
  real<lower=0> beta_1;
}

parameters {
  // Non-centered parameterization for HMC efficiency.
  // Per-player, using their individual prior scale (elementwise product below).
  vector[P] a_raw;
  vector[P] b_raw;
}

transformed parameters {
  // a[p] ~ Normal(prior_mu_a[p], prior_sigma_a[p]), constrained positive
  // Non-centered: a = mu + sigma * raw, where raw ~ std_normal()
  vector<lower=0>[P] a = prior_mu_a + prior_sigma_a .* a_raw;
  vector<lower=0>[P] b = prior_mu_b + prior_sigma_b .* b_raw;
}

model {
  a_raw ~ std_normal();
  b_raw ~ std_normal();

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
  // Ellipse area at tau_h = 5 seconds
  vector[P] ellipse_area_5s;
  for (p in 1:P)
    ellipse_area_5s[p] = pi() * a[p] * b[p] * 25.0;

  // Log-likelihood per observation (for LOO-CV)
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
