data {
  int<lower=1> N;   // total number of observations
  int<lower=1> J;   // total number of participants 
  int<lower=1, upper=J> id[N]; // participant ID for each observation
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
  real<lower=sigma_mod> sigma_med; // difference between sigmas
}

transformed parameters {
  vector<lower=0, upper=1>[N] theta; // weight 
  real inv_mse_med = 1 / (square(mu_med) + square(sigma_med)); 
  vector[N] inv_mse_mod;  

  for (n in 1:N) {
    inv_mse_mod[n] = 1 / (square(x_mod[n] - x_med[n] + mu_mod) + square(sigma_mod)); 

    theta[n] = inv_mse_med / (inv_mse_med + inv_mse_mod[n]); 
  }
}

model {
  // priors
  mu_mod_mean ~ normal(); 
  log_mu_mod_sd ~ normal(); 
  sigma_mod_mean ~ normal(); 
  log_sigma_mod_sd ~ normal(); 

  mu_med_mean ~ normal(); 
  log_mu_med_sd ~ normal(); 
  sigma_med_mean ~ normal(); 
  log_sigma_med_sd ~ normal(); 

  for (j in 1:J){
    mu_mod[j] ~ normal(mu_mod_mean, exp(log_mu_mod_sd));
    sigma_mod[j] ~ normal(sigma_mod_mean, exp(log_sigma_mod_sd)); 
    mu_med[j] ~ normal(mu_med_mean, exp(log_mu_med_sd)); 
    sigma_med[j] ~ normal(sigma_med_mean, exp(log_sigma_med_sd)); 
  }

  // likelihood
  for (n in 1:N) {
    j = id[n]; 
    target += normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j]))); 
  }
}
generated quantities {
  vector[N] log_lik; 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    j = id[n]; 
    log_lik[n] = normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j])));  


    y_rep[n] = normal_rng(theta[n] * (x_med[n] + mu_med[j]) + (1 - theta[n]) * (x_mod[n] + mu_mod[j]),
                                 sqrt(square(theta[n]) * square(sigma_med[j]) + square(1 - theta[n]) * square(sigma_mod[j])));  

  }
}
