library(tidyverse)
library(sgt)
library(ggplot2)

x <- seq(-5, 5, 0.01)
y <- dsgt(x, mu = 0, sigma = 1, 
          lambda = 0, p = 2, q = 10)
data.frame(x = x, y = y ) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() + 
  theme_minimal() + 
  xlim(-5, 5) +
  ylim(0, 1)

y <- psgt(x, mu = 0, sigma = 1, 
          lambda = 0, p = 2, q = 10)
data.frame(x = x, y = y ) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() + 
  theme_minimal() + 
  xlim(-5, 5) +
  ylim(0, 1)

dsgt(0.97, 0, 1, 0, 2, 10)
qsgt(0.5, 0, 1, 0.2, 2, 10)
