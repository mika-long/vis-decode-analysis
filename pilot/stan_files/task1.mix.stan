data {
  int<lower=1> N;              // total number of observations
  int<lower=1> J;              // number of participants
  int<lower=1, upper=J> id[N]; // participant ID for each observation
  vector[N] x;                 // observed values
  vector[N] x_med;             // known median values
  vector[N] x_mod;             // known mode values
  
  // Informative priors from brms fit
  real mu_mod_mean;            // Mean of participant intercepts (0.01)
  real<lower=0> mu_mod_sd;     // SD of participant intercepts (0.06)
  real log_sigma_mod_mean;     // Mean of log(sigma) (-1.30)
  real<lower=0> log_sigma_mod_sd; // SD of log(sigma) (0.20)
}

parameters {
  // Population-level parameters
  real mu_mod_pop;
  real mu_med_pop;
  real log_sigma_mod_pop;
  real log_sigma_med_pop;
  
  // Participant-level parameters (centered parameterization)
  vector[J] mu_mod;
  vector[J] mu_med;
  vector[J] log_sigma_mod;
  vector[J] log_sigma_med; 
}

transformed parameters {
  // Participant-specific parameters
  vector<lower=0>[J] sigma_mod;
  vector<lower=0>[J] sigma_med;
  
  for (j in 1:J) {
    // Transform log parameters to natural scale
    sigma_mod[j] = exp(log_sigma_mod[j]); 
    simga_med[j] = exp(log_sigma_med[j]); 
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
  mu_mod_pop ~ normal(mu_mod_mean, sigma_mod_sd);  // Informative prior based on brms fit
  log_sigma_mod_pop ~ normal(log_sigma_mod_mean, log_sigma_mod_sd);  // Informative prior from brms

  mu_med_pop ~ normal(0, 1);  
  log_sigma_med_pop ~ normal(0, 1);  
  
  // Centered parameterization - directly model the parameters with their hierarchical priors
  mu_mod ~ normal(mu_mod_pop, mu_mod_sd);
  mu_med ~ normal(mu_med_pop, mu_med_sd);  // Assuming you want to use the same SD
  log_sigma_mod ~ normal(log_sigma_mod_pop, log_sigma_med_sd);
  log_sigma_med ~ normal(log_sigma_med_pop, log_sigma_med_sd);  // Assuming same SD
  
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

    // Sample component indicator using w[n] as the probability
    int component = bernoulli_rng(w[n]);
    
    // Sample from the selected component
    if (component == 1) {
      // If component = 1, sample from the first component (median)
      y_rep[n] = normal_rng(x_med[n] + mu_med[j], sigma_med[j]);
    } else {
      // If component = 0, sample from the second component (mode)
      y_rep[n] = normal_rng(x_mod[n] + mu_mod[j], sigma_mod[j]);
    }
  }
}


