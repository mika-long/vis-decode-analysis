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
  mu_med ~ normal(0, 1);
  sigma_med ~ normal(0, 1);
  
  // likelihood
  for (n in 1:N) {
    target += normal_lpdf(x[n] | (x_med[n] + mu_med) * w[n] + (x_mod[n] + mu_mod) * (1 - w[n]), 
                                 square(w[n] * sigma_med) + square((1 - w[n]) * sigma_mod));
  }
}

generated quantities {
  vector[N] log_lik; // log likilihood of observed data given the model parameters 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(x[n] | (x_med[n] + mu_med) * w[n] + (x_mod[n] + mu_mod) * (1 - w[n]), 
                                 square(w[n] * sigma_med) + square((1 - w[n]) * sigma_mod));

   y_rep[n] = normal_rng((x_med[n] + mu_med) * w[n] + (x_mod[n] + mu_mod) * (1 - w[n]), 
                                 square(w[n] * sigma_med) + square((1 - w[n]) * sigma_mod));
  }
}
