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
  // tighter sigma priors no longer appear necessary
  // sigma_mod ~ normal(0.08, 0.04);
  sigma_mod ~ normal(0, 1);
  mu_med ~ normal(0, 1);
  // sigma_med ~ normal(0.5, 1);
  sigma_med ~ normal(0, 1);

  // likelihood
  for (n in 1:N) {
    // weighted average of two normals - roughly like anchoring plus adjustment?
    target += normal_lpdf(
      x[n] |
      lambda[n] * (x_med[n] + mu_med) + (1 - lambda[n]) * (x_mod[n] + mu_mod),
      sqrt((lambda[n] * sigma_med)^2 + ((1 - lambda[n]) * sigma_mod)^2)
    );
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;

  for (n in 1:N) {
    log_lik[n] = normal_lpdf(
      x[n] |
      lambda[n] * (x_med[n] + mu_med) + (1 - lambda[n]) * (x_mod[n] + mu_mod),
      sqrt(lambda[n] * sigma_med^2 + (1 - lambda[n]) * sigma_mod^2)
    );

    // Generate posterior predictive samples
    y_rep[n] = normal_rng(
      lambda[n] * (x_med[n] + mu_med) + (1 - lambda[n]) * (x_mod[n] + mu_mod),
      sqrt(lambda[n] * sigma_med^2 + (1 - lambda[n]) * sigma_mod^2)
    );
  }
}
