data {
  int<lower=1> N;               // total number of observations
  int<lower=1> J;               // number of participants
  int<lower=1, upper=J> id[N];  // participant ID for each observation
  vector[N] x;                  // observed values
  vector[N] x_med;              // known median values
  vector[N] x_mod;              // known mode values

  // For informative priors
  real mu_mod_prior_mean;
  real<lower=0> mu_mod_prior_sd;
  real<lower=0> sigma_mod_prior_mean;
  real<lower=0> sigma_mod_prior_sd;
}

parameters {
  // Population-level parameters
  real mu_mod_pop;
  real mu_med_pop;
  real<lower=0> sigma_mod_pop;
  real<lower=sigma_mod> sigma_med_pop;

  // Participant-level parameters (non-centered parameterization)
  vector[J] mu_mod_z;
  vector[J] mu_med_z;
  vector<lower=0>[J] sigma_mod_raw;
  vector<lower=0>[J] sigma_med_raw;
}


transformed parameters {
  // participant-specific paremeters
  vector<lower=0, upper=1>[N] w; // weight 

  real inv_MSE_med = 1 / (square(mu_med) + square(sigma_med)); 

  for (n in 1:N) {
    // Calculate difference between mode and median
    real inv_MSE_mod = 1 / (square(x_mod[n] - x_med[n] + mu_mod) + square(sigma_mod));

    w[n] = inv_MSE_med / (inv_MSE_med + inv_MSE_mod); 
  }
}

model {
  // priors
  // mu_mod ~ normal(0.01, 0.01);
  // sigma_mod ~ normal(0.15, 0.01);
  // Note that the above are fitted using Laplace
  // We have the following when fitting using Normal
  mu_mod ~ normal(-0.02, 0.02);
  sigma_mod ~ normal(0.23, 0.02);
  mu_med ~ normal(0, 0.05);
  sigma_med ~ normal(0.15, 0.6);
  
  // likelihood
  for (n in 1:N) {
    // Weighted average of two models 
    // target += w[n] * normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med) +
    //          (1 - w[n]) * normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod);
    // I don't think we can do simple weighted averages because things are no log scale and the weights are not in log scale ...
    target += log_mix(w[n],
                  normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                  normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));
  }
}

generated quantities {
  vector[N] log_lik; // log likilihood of observed data given the model parameters 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    //log_lik[n] = w[n] * normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med) +
    //          (1 - w[n]) * normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod);
    log_lik[n] = log_mix(w[n],
                  normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                  normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));

    // Generate posterior predictive samples
    real normal_sample = normal_rng(x_med[n] + mu_med, sigma_med);
    real laplace_sample = normal_rng(x_mod[n] + mu_mod, sigma_mod);

    // Weighted average using the observation-specific weight w[n]
    y_rep[n] = w[n] * normal_sample + (1 - w[n]) * laplace_sample;

    // Generate posterior predictive samples
    //if (bernoulli_rng(w[n]) == 1) {
    //   // Sample from normal component
    //   y_rep[n] = normal_rng(x_med[n] + mu_med, sigma_med);
    // } else {
    //   y_rep[n] = normal_rng(x_mod[n] + mu_mod, sigma_mod);
    // }
  }
}
