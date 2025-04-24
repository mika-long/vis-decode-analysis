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
  mu_mod ~ normal(0.01, 0.01);
  sigma_mod ~ normal(0.15, 0.01);
  mu_med ~ normal(0, 0.05);
  sigma_med ~ normal(0.15, 0.6);
  
  // likelihood
  for (n in 1:N) {
    // Two-component mixture model
    target += normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med) + (1 - theta[n]) * (x_mod[n] + mu_mod),
                                 sqrt(square(theta[n]) * square(sigma_med) + square(1 - theta[n]) * square(sigma_mod))); 
  }
}
generated quantities {
  vector[N] log_lik; 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(x[n] | theta[n] * (x_med[n] + mu_med) + (1 - theta[n]) * (x_mod[n] + mu_mod),
                                 sqrt(square(theta[n]) * square(sigma_med) + square(1 - theta[n]) * square(sigma_mod))); 


    y_rep[n] = normal_rng(theta[n] * (x_med[n] + mu_med) + (1 - theta[n]) * (x_mod[n] + mu_mod),
                                 sqrt(square(theta[n]) * square(sigma_med) + square(1 - theta[n]) * square(sigma_mod)));

  }
}
