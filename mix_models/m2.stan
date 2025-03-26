data {
  int<lower=1> N; // total number of observations
  vector[N] y; // observed values 
  vector[N] y_med; // known median values 
  vector[N] y_mod; // known mode values 
  
  // participant data 
  int<lower=1> N_p; // number of participants 
  array[N] int<lower=1> PID; // participant ID for each observation 
}

parameters {
  vector<lower=0, upper=1>[N_p] lambda;
  
  // participant-specific standard deviations that are ORDERED  
  vector<lower=0>[N_p] sigma_smaller; // will be sigma_mod 
  vector<lower=0>[N_p] sigma_diff; // difference between sigmas
}

transformed parameters {
  vector<lower=0>[N_p] sigma_mod; 
  vector<lower=0>[N_p] sigma_med;

  // calculate sigma_mod and sigma_med 
  for (p in 1:N_p) {
    sigma_mod[p] = sigma_smaller[p]; 
    sigma_med[p] = sigma_smaller[p] + sigma_diff[p]; // ensures sigma_med >= sigma_mod 
  }
}

model {
  // priors
  for (p in 1:N_p) {
    sigma_smaller[p] ~ normal(0.1, 0.2); 
    sigma_diff[p] ~ exponential(2); 
    lambda[p] ~ beta(2, 2);
  }
  
  // likelihood
  for (n in 1:N) {
    int p = PID[n]; // participant ID for this observation 
    
    // Two-component mixture model
    target += log_mix(lambda[p],
                      normal_lpdf(y[n] | y_med[n], sigma_med[p]), 
                      normal_lpdf(y[n] | y_mod[n], sigma_mod[p]));
  }
}

generated quantities {
  vector[N] log_lik; 
  
  for (n in 1:N) {
    int p = PID[n]; 
    
    log_lik[n] = log_mix(lambda[p],
                         normal_lpdf(y[n] | y_med[n], sigma_med[p]), 
                         normal_lpdf(y[n] | y_mod[n], sigma_mod[p]));
  }
}
