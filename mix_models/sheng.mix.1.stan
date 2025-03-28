data {
  int<lower=1> N; // total number of observations
  vector[N] x; // observed values
  vector[N] x_med; // known median values
  vector[N] x_mod; // known mode values
}

parameters {
  // mixture weight
  real<lower=0, upper=1> lambda;

  // intercept for x_med and x_mod
  real mu_mod;
  real mu_med;
  
  // standard deviations for x_med and x_mod
  real<lower=0> sigma_mod;
  real<lower=sigma_mod> sigma_med; // difference between sigmas
}

model {
  // priors
  mu_mod ~ normal(0.01, 0.01);
  sigma_mod ~ normal(0.15, 0.01);
  mu_med ~ normal(0, 0.05);
  sigma_med ~ normal(0.15, 0.6);
  lambda ~ beta(2, 2);
  
  // likelihood
  for (n in 1:N) {
    // Two-component mixture model
    target += log_mix(lambda,
                      normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                      double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));
  }
}
generated quantities {
  vector[N] log_lik; 
  vector[N] y_rep;
  
  for (n in 1:N) {
    log_lik[n] = log_mix(lambda,
                         normal_lpdf(x[n] | x_med + mu_med, sigma_med),
                         double_exponential_lpdf(x[n] | x_mod + mu_mod, sigma_mod));

    // Generate posterior predictive samples
    if (bernoulli_rng(lambda) == 1) {
      // Sample from normal component
      y_rep[n] = normal_rng(x_med[n] + mu_med, sigma_med);
    } else {
      y_rep[n] = double_exponential_rng(x_mod[n] + mu_mod, sigma_mod);
    }
  }
}
