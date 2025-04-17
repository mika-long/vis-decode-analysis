data {
  int<lower=1> N;   // total number of observations
  vector[N] x;      // observed values
  vector[N] x_med;  // known median values
  vector[N] x_mod;  // known mode values
}

parameters {
  // intercept for x_med and x_mod
  real mu_mod;
  real mu_med;

  // standard deviations for x_med and x_mod
  real<lower=0> sigma_mod;
  real<lower=0> sigma_med;
}

transformed parameters {
  vector<lower=0, upper=1>[N] lambda;
  real<lower=0> inverse_mse_med = 1 / square(sigma_med);
  vector<lower=0>[N] inverse_mse_mod = 1 / (square(x_med - x_mod) + square(sigma_mod));

  lambda = inverse_mse_med / (inverse_mse_med + inverse_mse_mod);
}

model {
  // priors
  mu_mod ~ normal(0, 1);
  sigma_mod ~ normal(0.08, 0.04);
  mu_med ~ normal(0, 1);
  sigma_med ~ normal(0.5, 1);

  // likelihood
  for (n in 1:N) {
    // Two-component mixture model
    target += log_mix(lambda[n],
                      normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                      double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;

  for (n in 1:N) {
    log_lik[n] = log_mix(lambda[n],
                         normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                         double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));

    // Generate posterior predictive samples
    if (bernoulli_rng(lambda[n]) == 1) {
      // Sample from normal component
      y_rep[n] = normal_rng(x_med[n] + mu_med, sigma_med);
    } else {
      y_rep[n] = double_exponential_rng(x_mod[n] + mu_mod, sigma_mod);
    }
  }
}
