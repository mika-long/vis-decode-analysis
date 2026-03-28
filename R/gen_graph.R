# This is the file for generating inset graphs
# used in the paper.

library(sgt)
library(ggplot2)
library(tidyverse)

##########
# Plot a new example as inset graphs -- Highest Point -- CDF Mode
##########
x <- seq(-5, 5, length.out = 100)
y <- dsgt(
  x,
  mu = -1.1,
  sigma = 0.9,
  lambda = -0.3,
  p = 2.6,
  q = 2.17,
  mean.cent = FALSE
)
x_val <- -1.1
y_val <- dsgt(
  x_val,
  mu = -1.1,
  sigma = 0.9,
  lambda = -0.3,
  p = 2.6,
  q = 2.17,
  mean.cent = FALSE
)
example_plot <- tibble(x = x, y = y) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  annotate("point", x = x_val, y = y_val, color = "red", size = 2) +
  coord_cartesian(xlim = c(-5, 5), ylim = c(0, 1)) +
  theme(
    axis.title.x = element_blank(), # Removes only the x-axis title
    axis.title.y = element_blank() # Removes only the y-axis title
  ) +
  tidybayes::theme_tidybayes()

example_plot

ggsave(
  here::here("figs", "HighestPoint_exmaple.svg"),
  example_plot,
  width = 3,
  height = 2.25,
  unit = "in",
  dpi = 300
)

##########
# Plot a new example as inset graphs -- Max Slope -- CDF Mode
##########

x <- seq(-5, 5, length.out = 100)
# cdf
y <- psgt(
  x,
  mu = 0.1,
  sigma = 1,
  lambda = 0.2,
  p = 2.6,
  q = 2.17,
  mean.cent = TRUE
)

# TODO -- we might need to turn this into a new supplementary material to explain how things work with generalized sgt ...
x_mode <- 0.1 # mean when mean.cent = FALSE
y_mode <- psgt(
  0.1,
  mu = 0.1,
  sigma = 1,
  lambda = 0.2,
  p = 2.6,
  q = 2.17,
  mean.cent = TRUE
)

example_plot <- tibble(x = x, y = y) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  annotate("point", x = x_mode, y = y_mode, color = "red", size = 2) +
  coord_cartesian(xlim = c(-5, 5), ylim = c(0, 1)) +
  theme(
    axis.title.x = element_blank(), # Removes only the x-axis title
    axis.title.y = element_blank() # Removes only the y-axis title
  ) +
  tidybayes::theme_tidybayes()

example_plot
ggsave(
  here::here("figs", "MaxSlope_exmaple.svg"),
  example_plot,
  width = 3,
  height = 2.25,
  unit = "in",
  dpi = 300
)

##########
# Plot a new example as inset graphs -- Bisect Area -- PDF Median
##########

x <- seq(-5, 5, length.out = 100)
# pdf
y <- dsgt(
  x,
  mu = -0.1,
  sigma = 1.6,
  lambda = 0.3,
  p = 2.6,
  q = 2.17,
  mean.cent = FALSE
)

x_median <- qsgt(
  0.5,
  mu = -0.1,
  sigma = 1.6,
  lambda = 0.3,
  p = 2.6,
  q = 2.17,
  mean.cent = FALSE
)
y_median <- dsgt(
  x_median,
  mu = -0.1,
  sigma = 1.6,
  lambda = 0.3,
  p = 2.6,
  q = 2.17,
  mean.cent = FALSE
)

example_plot <- tibble(x = x, y = y) %>%
  ggplot(aes(x = x, y = y)) +
  geom_line() +
  coord_cartesian(xlim = c(-5, 5), ylim = c(0, 1)) +
  annotate("point", x = x_median, y = y_median, size = 2, color = "red") +
  theme(
    axis.title.x = element_blank(), # Removes only the x-axis title
    axis.title.y = element_blank() # Removes only the y-axis title
  ) +
  tidybayes::theme_tidybayes()

example_plot
ggsave(
  here::here("figs", "BisectArea_example.svg"),
  example_plot,
  width = 3,
  height = 2.25,
  unit = "in",
  dpi = 300
)
