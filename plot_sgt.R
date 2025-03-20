library(sgt)
library(ggplot2)

x_vals <- seq(-5, 5, length.out = 1000)
pdf_vals <- dsgt(x_vals, mu=0.3, sigma=1, lambda=0.8, p=3, q=10, 
                 mean.cent = FALSE)
med <- qsgt(0.5, mu=0.3, sigma=1, lambda=0.8, p=3, q=10, 
            mean.cent = FALSE)

data.frame(x = x_vals, density = pdf_vals) %>% 
  ggplot(aes(x = x, y = density)) + 
  geom_line() + 
  theme_minimal() + 
  ylim(0, 1) + 
  geom_vline(xintercept = 0.3, linetype = "dashed", color = "red") + 
  geom_vline(xintercept = med, linetype = "dashed", color = "blue") + 
  labs(title = "lambda = 0.8")
