# origin is top left
# data_to_pixel_x <- function(data_x) {
#   return(53.5 * data_x + 317.5)
# }
# data_to_pixel_y <- function(data_y) {
#   return(-395 * data_y + 410)
# }

# tolerance for numerical precision
tolerance <- 1e-10

# --- Core Utilities ---
mm_to_visual_angle <- function(size_mm, distance) {
  2 * atan(size_mm / (2 * distance)) * 180 / pi
}

visual_angle_to_mm <- function(angle_deg, distance) {
  2 * distance * tan(angle_deg * pi / 180 / 2)
}

# --- Factory ---
make_va_scale <- function(data_min, data_max, px_min, px_max) {
  data_range <- data_max - data_min
  px_extent <- abs(px_max - px_min)

  # 1) data -> pixel and pixel -> data
  data_to_px <- function(data) {
    (data - data_min) / data_range * (px_max - px_min) + px_min
  }
  px_to_data <- function(px) {
    (px - px_min) / (px_max - px_min) * data_range + data_min
  }
  # 2) pixel -> mm and mm -> pixel
  px_to_phys <- function(px, px_per_mm) {
    abs(px - px_min) / px_per_mm
  }
  phys_to_px <- function(phys, px_per_mm) {
    sign(px_max - px_min) * phys * px_per_mm + px_min
  }

  # 3) mm -> vis angle
  phys_to_va <- function(size_mm, distance) {
    mm_to_visual_angle(size_mm, distance)
  }
  va_to_phys <- function(va, distance) {
    visual_angle_to_mm(va, distance)
  }

  # --- Composite functions ---
  data_to_phys <- function(data, px_per_mm) {
    data |> data_to_px() |> px_to_phys(px_per_mm)
  }
  phys_to_data <- function(phys, px_per_mm) {
    phys |> phys_to_px(px_per_mm) |> px_to_data()
  }

  data_to_va <- function(data, distance, px_per_mm) {
    data |> data_to_px() |> px_to_phys(px_per_mm) |> phys_to_va(distance)
  }
  va_to_data <- function(va, distance, px_per_mm) {
    va |> va_to_phys(distance) |> phys_to_px(px_per_mm) |> px_to_data()
  }
  px_to_va <- function(px, distance, px_per_mm) {
    px |> px_to_data() |> data_to_va(distance, px_per_mm)
  }

  list(
    data_to_px = data_to_px,
    px_to_data = px_to_data,
    data_to_phys = data_to_phys,
    phys_to_data = phys_to_data,
    data_to_va = data_to_va,
    va_to_data = va_to_data,
    px_to_va = px_to_va
  )
}

# # alt way
# data_to_pixel_x <- function(
#   data_x,
#   x_data_min = -5,
#   x_data_max = 5,
#   x_px_max = 585,
#   x_px_min = 50
# ) {
#   x_px_min +
#     (data_x - x_data_min) / (x_data_max - x_data_min) * (x_px_max - x_px_min)
# }
# data_to_pixel_y <- function(
#   data_y,
#   y_data_min = 0,
#   y_data_max = 1,
#   y_px_min = 15,
#   y_px_max = 410
# ) {
#   y_px_max -
#     (data_y - y_data_min) / (y_data_max - y_data_min) * (y_px_max - y_px_min)
# }

# # origin is bottom left
# pixel_to_phy_x <- function(pixel, pxMM, x_px_min = 50) {
#   (pixel - x_px_min) / pxMM
# }
# pixel_to_phy_y <- function(pixel, pxMM, y_px_max = 410) {
#   (y_px_max - pixel) / pxMM
# }

# # return visual angle in degrees and not radian
# vis_angle <- function(size, distance) {
#   2 * atan(size / (2 * distance)) * 180 / pi
# }
# # Inverse of vis_angle: visual angle (degrees) → physical size (mm)
# inv_vis_angle <- function(va, distToScreen) {
#   2 * distToScreen * tan(va * pi / 360)
# }

# # Inverse of pixel_to_phy_y: physical (mm) → pixel (top-left origin)
# phy_to_pixel_y <- function(phy, pxMM) {
#   410 - phy * pxMM
# }

# # Inverse of pixel_to_phy_x: physical (mm) → pixel
# phy_to_pixel_x <- function(phy, pxMM) {
#   phy * pxMM + 50
# }

# # Inverse of data_to_pixel_y: pixel → data
# pixel_to_data_y <- function(pixel) {
#   (410 - pixel) / 395
# }

# # Inverse of data_to_pixel_x: pixel → data
# pixel_to_data_x <- function(pixel) {
#   (pixel - 317.5) / 53.5
# }

# # ── Composed: va → numerical ────────────────────────────────
# va_to_numerical_y <- function(va, pxMM, distToScreen) {
#   phy <- inv_vis_angle()(va, distToScreen)
#   pixel <- phy_to_pixel_y(phy, pxMM)
#   pixel_to_data_y(pixel)
# }

# va_to_numerical_x <- function(va, pxMM, distToScreen) {
#   phy <- inv_vis_angle(va, distToScreen)
#   pixel <- phy_to_pixel_x(phy, pxMM)
#   pixel_to_data_x(pixel)
# }
