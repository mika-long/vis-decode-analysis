data_to_pixel_x <- function(data_x) {
  return(53.5 * data_x + 317.5)
}
data_to_pixel_y <- function(data_y) {
  return(-395 * data_y + 410)
}
pixel_to_phy_x <- function(pixel, pxMM, x_px_min = 50) {
  (pixel - x_px_min) / pxMM
}
pixel_to_phy_y <- function(pixel, pxMM, y_px_max = 410) {
  (y_px_max - pixel) / pxMM
}
vis_angle <- function(size, distance) {
  2 * atan(size / (2 * distance)) * 180 / pi
}
