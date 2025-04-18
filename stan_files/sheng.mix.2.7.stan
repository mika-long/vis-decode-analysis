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
  real<lower=sigma_mod> sigma_med; // could also model as difference between sigmas
}

transformed parameters {
  vector<lower=0, upper=1>[N] w; // weight 

  real inv_MSE_med = 1 / (square(mu_med) + square(sigma_med)); 

  for (n in 1:N) {
    // Calculate difference between mode and median
    real inv_MSE_mod = 1 / (square(x_mod[n] - x_med[n] + mu_mod) + 2 * square(sigma_mod)); 

    w[n] = inv_MSE_med / (inv_MSE_med + inv_MSE_mod); 
  }
}

model {
  // priors
  mu_mod ~ normal(0.01, 0.01);
  sigma_mod ~ normal(0.15, 0.01);
  mu_med ~ normal(0, 0.05);
  sigma_med ~ normal(0.15, 0.6);
  
  // likelihood
  for (n in 1:N) {
    // Weighted average of two models 
    // target += w[n] * normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med) +
    //          (1 - w[n]) * double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod);
    // I don't think we can do simple weighted averages because things are no log scale and the weights are not in log scale ...
    target += log_mix(w[n],
                  normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                  double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));
  }
}

generated quantities {
  vector[N] log_lik; // log likilihood of observed data given the model parameters 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    //log_lik[n] = w[n] * normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med) +
    //          (1 - w[n]) * double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod);
    log_lik[n] = log_mix(w[n],
                  normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                  double_exponential_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));

    // Generate posterior predictive samples
    real normal_sample = normal_rng(x_med[n] + mu_med, sigma_med);
    real laplace_sample = double_exponential_rng(x_mod[n] + mu_mod, sigma_mod);

    // Weighted average using the observation-specific weight w[n]
    y_rep[n] = w[n] * normal_sample + (1 - w[n]) * laplace_sample;

    // Generate posterior predictive samples
    //if (bernoulli_rng(w[n]) == 1) {
    //   // Sample from normal component
    //   y_rep[n] = normal_rng(x_med[n] + mu_med, sigma_med);
    // } else {
    //   y_rep[n] = double_exponential_rng(x_mod[n] + mu_mod, sigma_mod);
    // }
  }
}
