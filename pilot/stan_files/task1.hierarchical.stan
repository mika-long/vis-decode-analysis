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
  vector[J] mu_mod; 
  vector[J] mu_med; 
  vector<lower=0>[J] sigma_mod; 
  vector<lower=0>[J] sigma_med; 

  // Non-centered parameterization 
  mu_mod = mu_mod_pop + sigma_mod_pop * mu_mod_z; 
  mu_med = mu_med_pop + sigma_med_pop * mu_med_z; 

  for (j in 1:J){
    sigma_mod[j] = sigma_mod_pop * sigma_mod_raw[j]; 
    // ensure sigma_med > sigma_mod for each participant 
    sigma_med[j] = sigma_mod[j] + sigma_med_pop * sigma_med_raw[j]; 
  }

  // Weighte for each observation 
  vector<lower=0, upper=1>[N] w; 

  for (n in 1:N) {
    real inv_MSE_med = 1 / (square(mu_med[j]) + square(sigma_med[j])); 
    real inv_MSE_mod = 1 / (square(x_mod[n] - x_med[n] + mu_mod[j]) + square(sigma_mod[j]));

    w[n] = inv_MSE_med / (inv_MSE_med + inv_MSE_mod); 
  }
}

model {
  // Hyper-priors for population-level parameters 
  // TODO -- change this for the actual fitted data 
  mu_mod_pop ~ normal(-0.02, 0.02);
  sigma_mod_pop ~ normal(0.23, 0.02);
  mu_med_pop ~ normal(0, 0.05);
  sigma_med_pop ~ normal(0.15, 0.6);

  // Priors for participant-level parameters 
  mu_mod_z ~ normal(0, 1); 
  mu_med_z ~ normal(0, 1); 
  sigma_mod_raw ~ normal(0, 1); 
  sigma_med_raw ~ normal(0, 1); 
  
  // likelihood
  for (n in 1:N) {
    int j = id[n]; 
    target += log_mix(w[n],
                  normal_lpdf(x[n] | x_med[n] + mu_med[j], sigma_med[j]),
                  normal_lpdf(x[n] | x_mod[n] + mu_mod[j], sigma_mod[j]));
  }
}

generated quantities {
  vector[N] log_lik;  // log likilihood of observed data given the model parameters 
  vector[N] y_rep;    // posterior predictive checks
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
