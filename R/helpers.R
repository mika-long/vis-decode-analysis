# origin is top left  
data_to_pixel_y <- function(data_y) {
  return(-395 * data_y + 410)
}
data_to_pixel_x <- function(data_x) {
  return (53.5 * data_x + 317.5)
}

# origin is bottom left 
pixel_to_phy_x <- function(pixel, pxMM){
  (pixel - 50) / pxMM
}
pixel_to_phy_y <- function(pixel, pxMM){
  (410 - pixel) / pxMM
}

# return visual angle in degrees and not radian
vis_angle <- function(size, distance){
  return(2 * atan(size / (2 * distance)) * 180 / pi)
}

# tolerance for numerical precision 
tolerance <- 1e-10