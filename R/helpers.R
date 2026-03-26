# origin is top left
data_to_pixel_y <- function(data_y) {
  return(-395 * data_y + 410)
}
data_to_pixel_x <- function(data_x) {
  return(53.5 * data_x + 317.5)
}

# origin is bottom left
pixel_to_phy_x <- function(pixel, pxMM) {
  (pixel - 50) / pxMM
}
pixel_to_phy_y <- function(pixel, pxMM) {
  (410 - pixel) / pxMM
}

# return visual angle in degrees and not radian
vis_angle <- function(size, distance) {
  return(2 * atan(size / (2 * distance)) * 180 / pi)
}

# tolerance for numerical precision
tolerance <- 1e-10

# Inverse of vis_angle: visual angle (degrees) → physical size (mm)
phy_from_va <- function(va, distToScreen) {
  2 * distToScreen * tan(va * pi / 360)
}

# Inverse of pixel_to_phy_y: physical (mm) → pixel (top-left origin)
phy_to_pixel_y <- function(phy, pxMM) {
  410 - phy * pxMM
}

# Inverse of pixel_to_phy_x: physical (mm) → pixel
phy_to_pixel_x <- function(phy, pxMM) {
  phy * pxMM + 50
}

# Inverse of data_to_pixel_y: pixel → data
pixel_to_data_y <- function(pixel) {
  (410 - pixel) / 395
}

# Inverse of data_to_pixel_x: pixel → data
pixel_to_data_x <- function(pixel) {
  (pixel - 317.5) / 53.5
}

# ── Composed: va → numerical ────────────────────────────────
va_to_numerical_y <- function(va, pxMM, distToScreen) {
  phy <- phy_from_va(va, distToScreen)
  pixel <- phy_to_pixel_y(phy, pxMM)
  pixel_to_data_y(pixel)
}

va_to_numerical_x <- function(va, pxMM, distToScreen) {
  phy <- phy_from_va(va, distToScreen)
  pixel <- phy_to_pixel_x(phy, pxMM)
  pixel_to_data_x(pixel)
}
