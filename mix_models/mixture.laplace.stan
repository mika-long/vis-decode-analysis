data {
  int<lower=1> N;                       // number of observations
  vector[N] x_select;                   // observed selections
  vector[N] x_med;                      // median value for each observation
  vector[N] x_mod;                      // modal value for each observation

  // participant related 
  int<lower=1> J;                       // number of participants
  array[N] int<lower=1, upper=J> PID;   // participant ID for each observation
  
  // Priors for x_mod parameters (you mentioned you have good priors for these)
  real mu_mod_prior_mean;
  real<lower=0> mu_mod_prior_sd;
  real<lower=0> sigma_mod_prior_alpha;
  real<lower=0> sigma_mod_prior_beta;
}

parameters {
  // Mixing parameter
  real<lower=0, upper=1> theta;
  
  // Participant-level parameters
  vector[J] mu_med;               // mean deviation for median component by participant
  vector<lower=0>[J] sigma_med;   // sd for median component by participant
  vector[J] mu_mod;               // mean deviation for modal component by participant
  vector<lower=0>[J] sigma_mod;   // scale for Laplace component by participant
  
  // Hyperparameters for hierarchical structure
  real mu_med_global;             // global mean for mu_med
  real<lower=0> sigma_med_global; // global sd for mu_med
  real<lower=0> tau_sigma_med;    // scale for sigma_med
  
  // Latent variables (optional - see note in model block)
  // vector[N] hat_x_med;
  // vector[N] hat_x_mod;
}

model {
  // Hyperpriors
  mu_med_global ~ normal(0, 5);
  sigma_med_global ~ cauchy(0, 2.5);
  tau_sigma_med ~ gamma(2, 0.1);
  
  // Hierarchical priors for participant-level parameters
  mu_med ~ normal(mu_med_global, sigma_med_global);
  sigma_med ~ gamma(2, tau_sigma_med);
  
  // Hyperpriors for modal component 
  mu_mod_global ~ normal(0.01, 0.02); 
  sigma_mu_mod ~ normal(0.03, 0.02); 
  sigma_mod_global ~ normal(-1.79, 0.10); 
  sigma_sigma_mod ~ normal(0.17, 0.12); 

  // Hierarchical priors 
  mu_mod ~ normal(mu_mod_global, sigma_mu_mod);
  for (j in 1:J) {
    log_sigma_mod[j] ~ normal(sigma_mod_global, sigma_sigma_mod); 
  }
  
  // Prior for mixing parameter
  theta ~ beta(2, 2);  // weakly informative prior centered at 0.5
  
  // Likelihood using marginalization approach
  for (i in 1:N) {
    real log_lik_med;
    real log_lik_mod;
    
    // Log-likelihood for median component (Normal)
    log_lik_med = normal_lpdf(x_select[i] | x_med[i] + mu_med[PID[i]], sigma_med[PID[i]]);
    
    // Log-likelihood for modal component (Laplace)
    log_lik_mod = double_exponential_lpdf(x_select[i] | x_mod[i] + mu_mod[PID[i]], sigma_mod[PID[i]]);
    
    // Mixture log-likelihood using log-sum-exp trick for numerical stability
    target += log_sum_exp(log(theta) + log_lik_med, 
                          log1m(theta) + log_lik_mod);
  }
  
  /* Alternative approach with explicit latent variables:
  for (i in 1:N) {
    // Sample latent variables
    hat_x_med[i] ~ normal(x_med[i] + mu_med[PID[i]], sigma_med[PID[i]]);
    hat_x_mod[i] ~ double_exponential(x_mod[i] + mu_mod[PID[i]], sigma_mod[PID[i]]);
    
    // Mixture model for observations
    target += log_mix(theta,
                     normal_lpdf(x_select[i] | hat_x_med[i], 0.0001),
                     normal_lpdf(x_select[i] | hat_x_mod[i], 0.0001));
  }
  */
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
    log_lik[i] = log_sum_exp(log(theta) + log_lik_med, 
                            log1m(theta) + log_lik_mod);
    
    // Generate indicator for which component most likely generated each observation
    component_indicator[i] = bernoulli_rng(theta * exp(log_lik_med) / 
                                          (theta * exp(log_lik_med) + (1-theta) * exp(log_lik_mod)));
  }
}