data {
  int<lower=1> N;   // total number of observations
  int<lower=1> J;   // total number of participants 
  array[N] int<lower=1, upper=J> id; // participant ID for each observation 
  vector[N] x;      // observed values
  vector[N] x_med;  // known median values
  vector[N] x_mod;  // known mode values
}

parameters {
  // Population-level parameters 
  real mu_mod_mean;
  real<lower=0> mu_mod_sd; 
  real mu_med_mean;
  real<lower=0> mu_med_sd; 
  
  // standard deviations for x_med and x_mod
  real log_sigma_mod_mean; 
  real<lower=0> log_sigma_mod_sd; 
  real log_sigma_med_mean; 
  real<lower=0> log_sigma_med_sd;

  vector[J] mu_mod_z; 
  vector[J] mu_med_z; 
  vector[J] log_sigma_mod_z; 
  vector[J] log_sigma_med_z; 
}

transformed parameters {
  // Particiipant-level parameters
  vector[J] mu_mod; 
  vector<lower=0>[J] sigma_mod; 
  vector[J] mu_med; 
  vector<lower=0>[J] sigma_med; 
  
  vector<lower=0, upper=1>[N] theta; // weight 
  vector[N] inv_mse_med;
  vector[N] inv_mse_mod;  

  for (n in 1:N) {
    int j = id[n]; 
    inv_mse_med[n] = 1 / (square(mu_med[j]) + square(sigma_med[j])); 
    inv_mse_mod[n] = 1 / (square(x_mod[n] - x_med[n] + mu_mod[j]) + square(sigma_mod[j])); 

    theta[n] = inv_mse_med[n] / (inv_mse_med[n] + inv_mse_mod[n]); 
  }

  // Non-centered param 
  mu_mod = mu_mod_mean + mu_mod_z * mu_mod_sd; 
  sigma_mod = exp(log_sigma_mod_mean + log_sigma_mod_z * log_sigma_mod_sd); 
  mu_med = mu_med_mean + mu_med_z * mu_med_sd; 
  sigma_med = exp(log_sigma_med_mean + log_sigma_med_z * log_sigma_med_sd); 
}

model {
  // priors
  mu_mod_mean ~ normal(0.01, 0.02); 
  mu_mod_sd ~ normal(0.04, 0.03); 
  log_sigma_mod_mean ~ normal(-1.37, 0.10); 
  log_sigma_mod_sd ~ normal(0.34, 0.10); 

  mu_med_mean ~ normal(0, 1); 
  mu_med_sd ~ normal(0, 1); 
  log_sigma_med_mean ~ normal(0, 1); 
  log_sigma_med_sd ~ normal(0, 1); 

  
  mu_mod_z ~ std_normal(); 
  mu_med_z ~ std_normal(); 
  log_sigma_mod_z ~ std_normal(); 
  log_sigma_med_z ~ std_normal(); 

  
  

  // mu_mod[j] ~ normal(mu_mod_mean, mu_mod_sd);
  // sigma_mod[j] ~ lognormal(log_sigma_mod_mean, log_sigma_mod_sd); 
  // mu_med[j] ~ normal(mu_med_mean, mu_med_sd); 
  // sigma_med[j] ~ lognormal(log_sigma_med_mean, log_sigma_med_sd); 
  
  

  // likelihood
  for (n in 1:N) {
    int j = id[n]; 
    target += normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j]))); 
  }
}
generated quantities {
  vector[N] log_lik; 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    int j = id[n]; 
    log_lik[n] = normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j])));  


    y_rep[n] = normal_rng(theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j])));  

  }
}
