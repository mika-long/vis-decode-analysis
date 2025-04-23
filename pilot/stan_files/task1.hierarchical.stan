data {
  int<lower=1> N;              // total number of observations
  int<lower=1> J;              // number of participants
  int<lower=1, upper=J> id[N]; // participant ID for each observation
  vector[N] x;                 // observed values
  vector[N] x_med;             // known median values
  vector[N] x_mod;             // known mode values
  
  // Informative priors from brms fit
  real mu_pop_mean;            // Mean of participant intercepts (0.01)
  real<lower=0> mu_pop_sd;     // SD of participant intercepts (0.06)
  real log_sigma_pop_mean;     // Mean of log(sigma) (-1.30)
  real<lower=0> log_sigma_pop_sd; // SD of log(sigma) (0.20)
}

parameters {
  // Population-level parameters
  real mu_mod_pop;
  real mu_med_pop;
  real log_sigma_mod_pop;
  real log_sigma_med_pop;
  
  // Participant-level parameters (non-centered parameterization)
  vector[J] mu_mod_z;
  vector[J] mu_med_z;
  vector[J] log_sigma_mod_z;
  vector[J] log_sigma_med_z;
  
  // Difference between sigma_med and sigma_mod (to ensure sigma_med > sigma_mod)
  vector<lower=0>[J] sigma_diff;
}

transformed parameters {
  // Participant-specific parameters
  vector[J] mu_mod;
  vector[J] mu_med;
  vector<lower=0>[J] sigma_mod;
  vector<lower=0>[J] sigma_med;
  
  // Non-centered parameterization
  mu_mod = mu_mod_pop + mu_pop_sd * mu_mod_z;
  mu_med = mu_med_pop + mu_pop_sd * mu_med_z;
  
  for (j in 1:J) {
    // Transform log parameters to natural scale
    sigma_mod[j] = exp(log_sigma_mod_pop + log_sigma_pop_sd * log_sigma_mod_z[j]);
    sigma_med[j] = sigma_mod[j] + sigma_diff[j]; // Ensuring sigma_med > sigma_mod
  }
  
  // Weights for each observation
  vector<lower=0, upper=1>[N] w;
  
  for (n in 1:N) {
    int j = id[n];
    real inv_MSE_med = 1 / (square(mu_med[j]) + square(sigma_med[j])); 
    real inv_MSE_mod = 1 / (square(x_mod[n] - x_med[n] + mu_mod[j]) + square(sigma_mod[j]));
    
    w[n] = inv_MSE_med / (inv_MSE_med + inv_MSE_mod); 
  }
}

model {
  // Hyper-priors for population-level parameters
  mu_mod_pop ~ normal(mu_pop_mean, 0.1);  // Informative prior based on brms fit
  mu_med_pop ~ normal(mu_pop_mean, 0.1);  // Informative prior based on brms fit
  
  // Log-scale priors for sigma parameters
  log_sigma_mod_pop ~ normal(log_sigma_pop_mean, 0.1);  // Informative prior from brms
  log_sigma_med_pop ~ normal(log_sigma_pop_mean, 0.1);  // Using same prior for now
  
  // Standard normal priors for non-centered parameters
  mu_mod_z ~ normal(0, 1);
  mu_med_z ~ normal(0, 1);
  log_sigma_mod_z ~ normal(0, 1);
  log_sigma_med_z ~ normal(0, 1);
  
  // Prior for sigma difference
  sigma_diff ~ normal(0, 0.1);
  
  // Likelihood
  for (n in 1:N) {
    int j = id[n];
    target += log_mix(w[n],
                normal_lpdf(x[n] | x_med[n] + mu_med[j], sigma_med[j]),
                normal_lpdf(x[n] | x_mod[n] + mu_mod[j], sigma_mod[j]));
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_rep;
  vector[N] x_org = x;
  
  for (n in 1:N) {
    int j = id[n];
    
    log_lik[n] = log_mix(w[n],
                normal_lpdf(x[n] | x_med[n] + mu_med[j], sigma_med[j]),
                normal_lpdf(x[n] | x_mod[n] + mu_mod[j], sigma_mod[j]));

    // Generate posterior predictive samples
    real normal_sample = normal_rng(x_med[n] + mu_med[j], sigma_med[j]);
    real normal_sample2 = normal_rng(x_mod[n] + mu_mod[j], sigma_mod[j]);

    // Weighted average using the observation-specific weight w[n]
    y_rep[n] = w[n] * normal_sample + (1 - w[n]) * normal_sample2;
  }
}