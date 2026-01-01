source("fun.r")

# Function to generate points on a semicircle
points_on_semicircle <- function(point1, point2, num_points) {
  p1 <- as.numeric(point1)
  p2 <- as.numeric(point2)

  # Normalize the input points
  p1 <- p1 / sqrt(sum(p1^2))
  p2 <- p2 / sqrt(sum(p2^2))

  # Calculate the normal vector of the plane
  normal_vector <- c(
    p1[2] * p2[3] - p1[3] * p2[2],
    p1[3] * p2[1] - p1[1] * p2[3],
    p1[1] * p2[2] - p1[2] * p2[1]
  )

  # Check if collinear
  if (sqrt(sum(normal_vector^2)) < 1e-9) {
    warning("Input points are collinear with the origin. Cannot form a unique plane.")
    return(matrix(numeric(0), nrow = 0, ncol = 3))
  }
  normal_vector <- normal_vector / sqrt(sum(normal_vector^2))

  # Angles from 0 to pi
  angles <- seq(0, pi, length.out = num_points)

  # Test points
  cross_prod <- function(a, b) {
    c(a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1])
  }

  test1 <- p2 * cos(angles[2]) +
           cross_prod(normal_vector, p2) * sin(angles[2]) +
           normal_vector * sum(normal_vector * p2) * (1 - cos(angles[2]))
  test2 <- p2 * cos(angles[num_points - 1]) +
           cross_prod(normal_vector, p2) * sin(angles[num_points - 1]) +
           normal_vector * sum(normal_vector * p2) * (1 - cos(angles[num_points - 1]))

  semicircle_points <- list()

  # Check signs
  sign_test1_p1 <- sign(sum(test1 * p1))
  sign_test1_p2 <- sign(sum(test1 * p2))
  sign_test2_p1 <- sign(sum(test2 * p1))
  sign_test2_p2 <- sign(sum(test2 * p2))

  if (sign_test2_p1 == sign_test2_p2 && sign_test1_p1 == sign_test1_p2 &&
      sign_test1_p1 == sign_test2_p1 && sign_test1_p2 == sign_test2_p2) {
    for (angle in angles) {
      rotated_point <- p2 * cos(angle) +
                       cross_prod(normal_vector, p2) * sin(angle) +
                       normal_vector * sum(normal_vector * p2) * (1 - cos(angle))
      semicircle_points <- c(semicircle_points, list(rotated_point))
    }
  } else {
    for (angle in angles) {
      rotated_point <- p2 * cos(-angle) +
                       cross_prod(normal_vector, p2) * sin(-angle) +
                       normal_vector * sum(normal_vector * p2) * (1 - cos(-angle))
      semicircle_points <- c(semicircle_points, list(rotated_point))
    }
  }

  return(do.call(rbind, semicircle_points))
}

# Main outliers_count function
outliers_count <- function(data, weights = NULL, dist = "arc", a = 0.99, borderdist = "mean", res = 500) {
  data <- as.matrix(data)
  data_x <- data[, 1]
  data_y <- data[, 2]
  data_z <- data[, 3]

  if (is.null(weights)) {
    weights <- rep(1, nrow(data))
  }
  weights <- as.numeric(weights)

  # Calculate depths
  datadepth <- ahD(data, weights)

  # Border depth
  borderdepth <- median(datadepth)

  # Non-bag points
  nonbagpoints <- data[datadepth < borderdepth, , drop = FALSE]

  # Find mid (center)
  datadict <- split(data.frame(data), datadepth)
  if (length(datadict[[as.character(max(datadepth))]]) == 1) {
    mid <- as.numeric(datadict[[as.character(max(datadepth))]][1, ])
  } else {
    midmean <- colMeans(datadict[[as.character(max(datadepth))]])
    mid <- midmean / sqrt(sum(midmean^2))
  }

  # Generate sphere grid
  phi <- seq(0, pi, length.out = res)
  theta <- seq(0, 2 * pi, length.out = res)
  grid <- expand.grid(phi = phi, theta = theta)
  x_sphere <- sin(grid$phi) * cos(grid$theta)
  y_sphere <- sin(grid$phi) * sin(grid$theta)
  z_sphere <- cos(grid$phi)

  # Rotation matrix
  rotation_matrix <- ortho_matrix(mid)

  # Rotate sphere
  rotated_coords <- t(rotation_matrix %*% t(cbind(x_sphere, y_sphere, z_sphere)))
  x_sphere_rotated <- rotated_coords[, 1]
  y_sphere_rotated <- rotated_coords[, 2]
  z_sphere_rotated <- rotated_coords[, 3]

  # Reshape to matrix
  x_sphere_rotated <- matrix(x_sphere_rotated, nrow = res, ncol = res)
  y_sphere_rotated <- matrix(y_sphere_rotated, nrow = res, ncol = res)
  z_sphere_rotated <- matrix(z_sphere_rotated, nrow = res, ncol = res)

  # Surface colors (depth > borderdepth)
  surface_colors_data <- matrix(0, nrow = res, ncol = res)
  for (i in 1:res) {
    coords <- cbind(x_sphere_rotated[, i], y_sphere_rotated[, i], z_sphere_rotated[, i])
    depths <- ahD(data, weights, coords)
    surface_colors_data[, i] <- as.numeric(depths > borderdepth)
  }

  # Find borders
  borders <- list()
  for (i in 1:res) {
    zero_indices <- which(surface_colors_data[, i] == 0)
    if (length(zero_indices) > 0) {
      border_idx <- zero_indices[1] - 1
      if (border_idx > 0) {
        borders <- c(borders, list(c(border_idx, i)))
      }
    }
  }

  # Calculate distances from mid
  middist <- matrix(0, nrow = res, ncol = res)
  for (i in 1:res) {
    for (j in 1:res) {
      point <- c(x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j])
      if (dist == "arc") {
        middist[i, j] <- arcdist(point, mid)
      } else if (dist == "cos") {
        middist[i, j] <- cosdist(point, mid)
      } else if (dist == "chord") {
        middist[i, j] <- chorddist(point, mid)
      } else {
        stop("Wrong format of dist")
      }
    }
  }

  # Border distance
  border_distances <- sapply(borders, function(b) middist[b[1], b[2]])
  if (borderdist == "max") {
    borderd <- max(border_distances)
  } else if (borderdist == "mean") {
    borderd <- mean(border_distances)
  } else {
    stop("Wrong format of borderdist")
  }

  # Factor
  if (dist == "arc") {
    factor <- arcMF(kappaarc(borderd), a)
  } else if (dist == "cos") {
    factor <- cosMF(kappacos(borderd), a)
  } else if (dist == "chord") {
    factor <- chordMF(kappachord(borderd), a)
  } else {
    stop("Wrong format of dist")
  }

  # Count outliers
  num_outliers <- 0
  for (k in 1:nrow(nonbagpoints)) {
    point <- nonbagpoints[k, ]
    grid <- points_on_semicircle(point, mid, res)
    if (nrow(grid) == 0) next
    depths <- ahD(data, weights, grid)
    bagornot <- as.numeric(depths >= borderdepth)
    borderindex <- which(bagornot == 0)[1] - 1
    if (is.na(borderindex) || borderindex < 1) next

    if (dist == "arc") {
      borderd_point <- arcdist(grid[borderindex, ], mid)
      loopd <- factor * borderd_point
      pointdist <- arcdist(point, mid)
    } else if (dist == "cos") {
      borderd_point <- cosdist(grid[borderindex, ], mid)
      loopd <- factor * borderd_point
      pointdist <- cosdist(point, mid)
    } else if (dist == "chord") {
      loopd <- factor * borderd
      pointdist <- chorddist(point, mid)
    } else {
      stop("Wrong format of dist")
    }

    if (pointdist > loopd) {
      num_outliers <- num_outliers + 1
    }
  }

  return(num_outliers)
}
