data {
  int<lower=1> N;                       // number of observations
  vector[N] x_select;                   // observed selections
  vector[N] x_med;                      // median value for each observation
  vector[N] x_mod;                      // modal value for each observation

  // participant related 
  int<lower=1> J;                       // number of participants
  array[N] int<lower=1, upper=J> PID;   // participant ID for each observation
}

parameters {
  // Mixing parameter
  // Hyperparameters for theta 
  real logit_intercept_theta; 
  real<lower=0> var_logit_theta; 
  // Raw deviations for non-centered parameterization 
  vector[J] z_theta; 
  
  // Participant-level parameters
  vector[J] mu_med;               // mean deviation for median component by participant
  vector[J] log_sigma_med;        // sd for median component by participant
  vector[J] mu_mod;               // mean deviation for modal component by participant
  vector[J] log_sigma_mod; 
  
  // Hyperparameters for hierarchical structure
  real intercept_mu_mod;          
  real<lower=0> var_mu_mod; 
  
  real intercept_sigma_mod; 
  real<lower=0> var_sigma_mod; 

  real intercept_mu_med;          // global mean for mu_med
  real<lower=var_mu_mod> var_mu_med;       // global sd for mu_med
  
  real intercept_sigma_med;       // global mean for sigma_med
  real<lower=var_sigma_mod> var_sigma_med;       // global sd for sigma_med 
}

transformed parameters {
  vector<lower=0>[J] sigma_med = exp(log_sigma_med); 
  vector<lower=0>[J] sigma_mod = exp(log_sigma_mod); 

  // compute participant-specific thetas 
  vector<lower=0, upper=1>[J] theta; 
  for (j in 1:J) {
    theta[j] = inv_logit(logit_intercept_theta + z_theta[j] * sqrt(var_logit_theta)); 
  }
}

model {
  // Hyperpriors
  /* TODO */ 
  intercept_mu_med ~ normal(0, 0.05);
  var_mu_med ~ cauchy(0, 2.5);
  intercept_sigma_med ~ normal(-2, 0.5);
  var_sigma_med ~ normal(0.5, 0.3); 
  /* END OF TODO */
  
  // Hierarchical priors for participant-level parameters
  mu_med ~ normal(intercept_mu_med, var_mu_med);
  log_sigma_med ~ normal(intercept_sigma_med, var_sigma_med); 
  
  // Hyperpriors for modal component 
  // Obtained from task 2 
  intercept_mu_mod ~ normal(0.01, 0.02); 
  var_mu_mod ~ normal(0.03, 0.02); 
  intercept_sigma_mod ~ normal(-1.79, 0.10); 
  var_sigma_mod ~ normal(0.17, 0.12); 

  // Hierarchical priors 
  mu_mod ~ normal(intercept_mu_mod, var_mu_mod);
  log_sigma_mod ~ normal(intercept_sigma_mod, var_sigma_mod); 
  
  // Prior for mixing parameter
  /* theta ~ beta(2, 2);  // weakly informative prior centered at 0.5 */ 
  logit_intercept_theta ~ normal(0, 1.5); 
  var_logit_theta ~ normal(0, 1); 
  z_theta ~ std_normal(); 

  
  // Likelihood using marginalization approach
  for (i in 1:N) {
    real log_lik_med;
    real log_lik_mod;
    
    // Log-likelihood for median component (Normal)
    log_lik_med = normal_lpdf(x_select[i] | x_med[i] + mu_med[PID[i]], sigma_med[PID[i]]);
    
    // Log-likelihood for modal component (Laplace)
    log_lik_mod = double_exponential_lpdf(x_select[i] | x_mod[i] + mu_mod[PID[i]], sigma_mod[PID[i]]);
    
    // Mixture log-likelihood using log-sum-exp trick for numerical stability
    target += log_sum_exp(log(theta[PID[i]]) + log_lik_med, 
                          log1m(theta[PID[i]]) + log_lik_mod);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] hat_x_med;
  vector[N] hat_x_mod;
  vector[N] component_indicator;
  
  for (i in 1:N) {
    real log_lik_med;
    real log_lik_mod;
    
    // Generate the latent variables
    hat_x_med[i] = normal_rng(x_med[i] + mu_med[PID[i]], sigma_med[PID[i]]);
    hat_x_mod[i] = double_exponential_rng(x_mod[i] + mu_mod[PID[i]], sigma_mod[PID[i]]);
    
    // Calculate log likelihoods for each component
    log_lik_med = normal_lpdf(x_select[i] | x_med[i] + mu_med[PID[i]], sigma_med[PID[i]]);
    log_lik_mod = double_exponential_lpdf(x_select[i] | x_mod[i] + mu_mod[PID[i]], sigma_mod[PID[i]]);
    
    // Store overall log likelihood
    log_lik[i] = log_sum_exp(log(theta[PID[i]]) + log_lik_med, 
                            log1m(theta[PID[i]]) + log_lik_mod);
    
    // Generate indicator for which component most likely generated each observation
    component_indicator[i] = bernoulli_rng(theta[PID[i]] * exp(log_lik_med) / 
                                          (theta[PID[i]] * exp(log_lik_med) + (1-theta[PID[i]]) * exp(log_lik_mod)));
  }
}
