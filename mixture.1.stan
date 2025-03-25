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
  real<lower=0, upper=1> lambda; // mixing weight
  
  // participant-specific standard deviations 
  vector<lower=0>[N_p] sigma_med; 
  vector<lower=0>[N_p] sigma_mod; 
}
model {
  // priors
  lambda ~ beta(2, 2); 
  sigma_med ~ student_t(3, 0, 0.5); // TODO --- change this 
  sigma_mod ~ student_t(3, 0, 0.5); // TODO --- change this 
  
  // likelihood
  for (n in 1:N) {
    int p = PID[n]; // participant ID for this observation 
    
    // Two-component mixture model
    target += log_mix(lambda, 
                      normal_lpdf(y[n] | y_med[n], sigma_med[p]), 
                      normal_lpdf(y[n] | y_mod[n], sigma_mod[p]));
  }
}
generated quantities {
  vector[N] log_lik; 
  
  for (n in 1:N) {
    int p = PID[n]; 
    
    log_lik[n] = log_mix(lambda, 
                         normal_lpdf(y[n] | y_med[n], sigma_med[p]), 
                         normal_lpdf(y[n] | y_mod[n], sigma_mod[p]));
  }
}
