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
  vector<lower=0, upper=1>[N] theta; // weight 

  for (n in 1:N) {
    real diff = x_med[n] - x_mod[n];
    real sum_var = sqrt(square(sigma_med) + square(sigma_mod));

    theta[n] = inv_logit(alpha + beta * (abs(diff) / sum_var));
  }
}

model {
  // priors
  mu_mod ~ normal(-0.02, 0.02); // Different prior than the laplace model 
  sigma_mod ~ normal(0.23, 0.02);
  mu_med ~ normal(0, 1);
  sigma_med ~ normal(0.15, 1);
  
  // likelihood
  for (n in 1:N) {
    // Two-component mixture model
    target += log_mix(theta[n],
                      normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                      normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));
  }
}
generated quantities {
  vector[N] log_lik; 
  vector[N] y_rep; // posterior predictive checks
  vector[N] x_org = x;
  
  for (n in 1:N) {
    log_lik[n] = log_mix(theta[n],
                         normal_lpdf(x[n] | x_med[n] + mu_med, sigma_med),
                         normal_lpdf(x[n] | x_mod[n] + mu_mod, sigma_mod));

    // Generate posterior predictive samples
    if (bernoulli_rng(theta[n]) == 1) {
      // Sample from normal component
      y_rep[n] = normal_rng(x_med[n] + mu_med, sigma_med);
    } else {
      y_rep[n] = normal_rng(x_mod[n] + mu_mod, sigma_mod);
    }
  }
}
