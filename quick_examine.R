library(tidyverse)
library(ggplot2)
library(sgt)

# Functions to convert between pixel coordinates and data coordinates in an SVG

#' Create a mapping object that contains the necessary information for coordinate transformations
#'
#' @param width Total width of the SVG in pixels
#' @param height Total height of the SVG in pixels
#' @param margins List with top, right, bottom, left margins in pixels
#' @param xDomain Vector with min and max values of the x data domain
#' @param yDomain Vector with min and max values of the y data domain
#' @return A list with mapping information and conversion functions
#' 
create_coordinate_mapper <- function(width, height, margins, xDomain, yDomain) {
  # Calculate the plotting area dimensions
  plot_width <- width - margins$left - margins$right
  plot_height <- height - margins$top - margins$bottom
  
  # Create scaling factors
  x_scale <- plot_width / (xDomain[2] - xDomain[1])
  y_scale <- plot_height / (yDomain[2] - yDomain[1])
  
  # Return a list with all the necessary information
  return(list(
    width = width,
    height = height,
    margins = margins,
    plot_width = plot_width,
    plot_height = plot_height,
    xDomain = xDomain,
    yDomain = yDomain,
    x_scale = x_scale,
    y_scale = y_scale
  ))
}

#' Convert data coordinates to pixel coordinates
#'
#' @param mapper The mapping object created by create_coordinate_mapper()
#' @param x The x data coordinate to convert
#' @param y The y data coordinate to convert
#' @return A list with x and y pixel coordinates
#' 
data_to_pixel <- function(mapper, x, y) {
  # Convert x data coordinate to pixel coordinate
  pixel_x <- mapper$margins$left + (x - mapper$xDomain[1]) * mapper$x_scale
  
  # Convert y data coordinate to pixel coordinate
  # Note: SVG y-axis is inverted (0 at top, increases downward)
  pixel_y <- mapper$height - mapper$margins$bottom - (y - mapper$yDomain[1]) * mapper$y_scale
  
  return(list(x = pixel_x, y = pixel_y))
}

#' Convert pixel coordinates to data coordinates
#'
#' @param mapper The mapping object created by create_coordinate_mapper()
#' @param pixel_x The x pixel coordinate to convert
#' @param pixel_y The y pixel coordinate to convert
#' @return A list with x and y data coordinates
#' 
pixel_to_data <- function(mapper, pixel_x, pixel_y) {
  # Convert x pixel coordinate to data coordinate
  data_x <- mapper$xDomain[1] + (pixel_x - mapper$margins$left) / mapper$x_scale
  
  # Convert y pixel coordinate to data coordinate
  # Note: SVG y-axis is inverted (0 at top, increases downward)
  data_y <- mapper$yDomain[1] + (mapper$height - mapper$margins$bottom - pixel_y) / mapper$y_scale
  
  return(list(x = data_x, y = data_y))
}

# Example usage:
# Define SVG dimensions and margins
svg_width <- 600
svg_height <- 450
svg_margins <- list(top = 15, right = 15, bottom = 40, left = 50)

# Define data domains
x_domain <- c(-5, 5)
y_domain <- c(0, 1)
# 
# # Create the mapper
mapper <- create_coordinate_mapper(svg_width, svg_height, svg_margins, x_domain, y_domain)
# 
# # Convert data coordinates to pixel coordinates
pixel_coords <- data_to_pixel(mapper, 0, 0.5)
print(pixel_coords)
# 
# # Convert pixel coordinates to data coordinates
# data_coords <- pixel_to_data(mapper, 400, 300)
# print(data_coords)

df_slider <- read.csv("vis-decode-slider_all_tidy.csv") %>% 
  as_tibble(.)

id_slider <- "7351a20a-fadb-41f4-bd2f-66dbe5a952d5"

df1 <- df_slider %>% filter(participantId == id_slider)

df1 %>% select(trialId, responseId, answer) %>% 
  filter(!grepl("task5", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  pivot_wider(names_from = responseId, values_from = answer, names_repair = "universal") %>% 
  separate_wider_delim(trialId, "_", names = c("TaskType", "type", "trial_id")) %>% 
  select(-type) -> p_df

p_df %>% mutate(median.x = case_when(TaskType == "task1" | TaskType == "task2" ~ qsgt(0.5, mu = param.mu, lambda = param.lambda, 
                                                                                      sigma = param.sigma, p = param.p, q = param.q), 
                                     .default = qsgt(0.5, mu = param.mu, lambda = param.lambda, 
                                                     sigma = param.sigma, p = param.p, q = param.q)), 
                median.y = case_when(TaskType == "task1" | TaskType == "task2" ~ dsgt(median.x, mu = param.mu, lambda = param.lambda, 
                                                                                      sigma = param.sigma, p = param.p, q = param.q), 
                                     .default = 0.5), 
                mode.x = case_when(TaskType == "task1" | TaskType == "task2" ~ param.mu, 
                                   .default = param.mu), 
                mode.y = case_when(TaskType == "task1" | TaskType == "task2" ~ dsgt(mode.x, mu = param.mu, lambda = param.lambda, 
                                                                                    sigma = param.sigma, p = param.p, q = param.q), 
                                   .default = psgt(mode.x, mu = param.mu, lambda = param.lambda, 
                                                   sigma = param.sigma, p = param.p, q = param.q))) -> full_df
full_df



p_df %>% select(contains("param")) %>% slice(1) %>% unlist(.) %>% as.numeric(.) -> t
dsgt(0, t[1], t[2], t[3], t[4], mean.cent = F)


df_slider %>% filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(!grepl("param", responseId)) %>% 
  select(participantId, trialId, responseId, answer) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `location-x`, y = `location-y`)) + 
  theme_minimal() + 
  geom_point() + 
  geom_hline(yintercept = 0.5, color="gray", linetype="dashed") + 
  ylim(0.45, 0.55)

df_slider %>% filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(!grepl("param", responseId)) %>% 
  select(participantId, trialId, responseId, answer) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  mutate(answer = answer / 3.27) %>% # pixel to mm ratio 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
  theme_minimal() + 
  geom_point() + 
  geom_hline(yintercept = 212.5 / 3.27, color="gray", linetype="dashed") +
  xlab("X (mm)") + ylab("Y (mm)") + 
  ylim(60, 70)

# df_slider %>% filter(grepl("task3", trialId)) %>% 
#   filter(grepl("test", trialId)) %>% 
#   filter(!grepl("param", responseId)) %>% 
#   select(participantId, trialId, responseId, answer) %>% 
#   mutate(answer = as.numeric(answer)) %>% 
#   pivot_wider(names_from = responseId, values_from = answer) %>% 
#   ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
#   theme_minimal() + 
#   geom_point() + 
#   geom_hline(yintercept = 212.5, color="gray", linetype="dashed")



#### CLICK 

df_click %>% filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(!grepl("param", responseId)) %>% 
  select(participantId, trialId, responseId, answer) %>% 
  filter(responseId %in% c('location-x', 'location-y')) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `location-x`, y = `location-y`)) + 
  theme_minimal() + 
  geom_point() + 
  geom_hline(yintercept = 0.5, color="gray", linetype="dashed") + 
  ylim(0.45, 0.55)

# df_slider %>% filter(grepl("task3", trialId)) %>% 
#   filter(grepl("test", trialId)) %>% 
#   filter(!grepl("param", responseId)) %>% 
#   select(participantId, trialId, responseId, answer) %>% 
#   mutate(answer = as.numeric(answer)) %>% 
#   pivot_wider(names_from = responseId, values_from = answer) %>% 
#   ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
#   theme_minimal() + 
#   geom_point() + 
#   geom_hline(yintercept = 212.5, color="gray", linetype="dashed")
# 
# df_slider %>% filter(grepl("task3", trialId)) %>% 
#   filter(grepl("test", trialId)) %>% 
#   filter(!grepl("param", responseId)) %>% 
#   select(participantId, trialId, responseId, answer) %>% 
#   mutate(answer = as.numeric(answer)) %>% 
#   mutate(answer = answer / 3.27) %>% # pixel to mm ratio 
#   pivot_wider(names_from = responseId, values_from = answer) %>% 
#   ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
#   theme_minimal() + 
#   geom_point() + 
#   geom_hline(yintercept = 212.5 / 3.27, color="gray", linetype="dashed") +
#   xlab("X (mm)") + ylab("Y (mm)")

#################

# let's only look at task 3 because this is the one for 0.5 
df1 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("location-x", "location-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  mutate(answer = case_when(
    responseId == 'location-x' ~ 5 + answer, 
    responseId == 'location-y' ~ answer
  )) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `location-x`, y = `location-y`)) + 
  theme_minimal() + 
  geom_point() + 
  xlab("Distance from y=0") + 
  ylim(0.47, 0.53) -> p1

df1 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("pixel-x", "pixel-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  # mutate(answer = case_when(
  #   responseId == 'location-x' ~ 5 + answer, 
  #   responseId == 'location-y' ~ answer
  # )) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
  theme_minimal() + 
  geom_hline(yintercept = 212.5, color="gray", linetype="dashed") +
  geom_point() -> p2

df1 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("pixel-x", "pixel-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  mutate(answer = answer / 3.27) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
  theme_minimal() + 
  geom_hline(yintercept = 212.5 / 3.27, color="gray", linetype="dashed") +
  geom_point() + 
  xlab("X (mm)") + 
  ylab("Y (mm)") -> p3


###### 
# let's only look at task 3 because this is the one for 0.5 
df2 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("location-x", "location-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  mutate(answer = case_when(
    responseId == 'location-x' ~ 5 + answer, 
    responseId == 'location-y' ~ answer
  )) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `location-x`, y = `location-y`)) + 
  theme_minimal() + 
  geom_point() + 
  ylim(0.47, 0.53) -> p4

df2 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("pixel-x", "pixel-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
  theme_minimal() + geom_point() # + 
  # geom_hline(yintercept = 212.5, color="gray", linetype="dashed") +
  geom_point() 

df2 %>% select(trialId, responseId, responsePrompt, answer) %>% 
  filter(grepl("task3", trialId)) %>% 
  filter(grepl("test", trialId)) %>% 
  filter(responseId %in% c("pixel-x", "pixel-y")) %>% 
  mutate(answer = as.numeric(answer)) %>% 
  select(trialId, responseId, answer) %>% 
  mutate(answer = answer / 3.27) %>% 
  pivot_wider(names_from = responseId, values_from = answer) %>% 
  ggplot(aes(x = `pixel-x`, y = `pixel-y`)) + 
  theme_minimal() + 
  geom_hline(yintercept = 212.5 / 3.27, color="gray", linetype="dashed") +
  geom_point() + 
  xlab("X (mm)") + 
  ylab("Y (mm)") 