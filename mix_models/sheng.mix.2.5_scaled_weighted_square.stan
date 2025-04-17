data {
  int<lower=1> N;   // total number of observations
  vector[N] x;      // observed values
  vector[N] x_med;  // known median values
  vector[N] x_mod;  // known mode values
}

parameters {
  // scaling factor for distance
  real beta;
  // intercept for distance
  real alpha;

  // intercept for x_med and x_mod
  real mu_mod;
  real mu_med;

  // standard deviations for x_med and x_mod
  real<lower=0> sigma_mod;
  real<lower=sigma_mod> sigma_med; // difference between sigmas
}

transformed parameters {
  vector<lower=0, upper=1>[N] lambda;
  real scale = 1/(square(sigma_med) + square(sigma_mod));

  lambda = inv_logit(beta * (square(x_med - x_mod)*scale - alpha));
}

model {
  // priors
  mu_mod ~ normal(0, 1);
  sigma_mod ~ normal(0.08, 0.04);
  mu_med ~ normal(0, 1);
  sigma_med ~ normal(0.5, 1);

  beta ~ normal(5, 0.5);
  alpha ~ normal(1, 0.4);

  // likelihood
  for (n in 1:N) {
    // weighted average of two normals - roughly like anchoring plus adjustment?
    target += normal_lpdf(
      x[n] |
      lambda[n] * (x_med[n] + mu_med) + (1 - lambda[n]) * (x_mod[n] + mu_mod),
      lambda[n] * sigma_med + (1 - lambda[n]) * sigma_mod
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
      lambda[n] * sigma_med + (1 - lambda[n]) * sigma_mod
    );

    // Generate posterior predictive samples
    y_rep[n] = normal_rng(
      lambda[n] * (x_med[n] + mu_med) + (1 - lambda[n]) * (x_mod[n] + mu_mod),
      lambda[n] * sigma_med + (1 - lambda[n]) * sigma_mod
    );
  }
}
