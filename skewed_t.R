# apparently there's skewed t and generalized skewed t ... 
library(sn)
# https://cran.r-project.org/web/packages/sn/index.html
# https://cran.r-project.org/web/packages/sn/vignettes/how_to_sample.pdf
library(sgt)
# https://cran.r-project.org/web/packages/sgt/
# https://cran.r-project.org/web/packages/sgt/sgt.pdf

x_values <- seq(-5, 5, by = 0.01)

# mu = 0
# sigma = 1, lambda = 0, p = 2, q = Inf

pdf_values <- dsgt(x_values, mu = 0, sigma = 1, lambda = 0, p = 10, q = 10)

# Create a data frame for plotting
plot_data <- data.frame(x = x_values, density = pdf_values)

# Create the plot
ggplot(plot_data, aes(x = x, y = density)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Generalized Skewed t-Distribution PDF",
       x = "x",
       y = "Density") +
  theme_minimal() + 
  ylim(0, 1) + xlim(-5, 5)


cdf_values <- psgt(x_values)

# Calculate CDF values using the psgt function
cdf_values <- psgt(x_values)

# Create a data frame for plotting
plot_data <- data.frame(x = x_values, probability = cdf_values)

# Create the plot
ggplot(plot_data, aes(x = x, y = probability)) +
  geom_line(color = "red", linewidth = 1) +
  labs(title = "Generalized Skewed t-Distribution CDF",
       x = "x",
       y = "Cumulative Probability") +
  theme_minimal()

# Define the same test cases as in JavaScript
test_cases <- list(
  list(x = 0, mu = 0, sigma = 1, lambda = 0, p = 2, q = 10),
  list(x = 1, mu = 0, sigma = 1, lambda = 0.5, p = 2, q = 5),
  list(x = -1, mu = 0, sigma = 1, lambda = -0.5, p = 2, q = 5),
  list(x = 2, mu = 1, sigma = 2, lambda = 0.3, p = 3, q = 7)
)

# Calculate dsgt values
for (test in test_cases) {
  r_result <- dsgt(test$x, mu = test$mu, sigma = test$sigma, lambda = test$lambda, p = test$p, q = test$q)
  cat(sprintf("x: %s, params: {mu: %s, sigma: %s, lambda: %s, p: %s, q: %s}, result: %s\n", 
              test$x, test$mu, test$sigma, test$lambda, test$p, test$q, r_result))
}
