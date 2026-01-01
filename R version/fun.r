# Install and load required packages
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools", repos = "http://cran.us.r-project.org")
}
devtools::install_github("DyckerhoffRainer/sphericalDepth")
library(devtools)
library(sphericalDepth)

# Defining function t, this will be used later to estimate the parameter kappa
t_func <- function(k, a = 0.99) {
  if (k != 0) {
    return ((log((1 - a) * exp(2 * k) + a) - k) / k)
  } else {
    return (1 - 2 * a)
  }
}

# Definitions of distance metrics
arcdist <- function(x, y) {
  acos(sum(x * y))
}

cosdist <- function(x, y) {
  1 - sum(x * y)
}

chorddist <- function(x, y) {
  sqrt(2 * (1 - sum(x * y)))
}

# Definition of the equations to be solved to estimate kappa
arceq <- function(k, d) {
  acos(t_func(k, a = 0.5)) - d
}

coseq <- function(k, d) {
  1 - t_func(k, a = 0.5) - d
}

chordeq <- function(k, d) {
  sqrt(2 * (1 - t_func(k, a = 0.5))) - d
}

# Defining functions for the estimation of kappa
kappaarc <- function(d) {
  uniroot(arceq, c(-10, 10), d = d)$root
}

kappacos <- function(d) {
  uniroot(coseq, c(-10, 10), d = d)$root
}

kappachord <- function(d) {
  uniroot(chordeq, c(-10, 10), d = d)$root
}

# Defining functions that calculate the multiplying factor, given the estimate of kappa
arcMF <- function(k, a = 0.99) {
  acos(t_func(k, a)) / acos(t_func(k, 0.5))
}

cosMF <- function(k, a = 0.99) {
  (1 - t_func(k, a)) / (1 - t_func(k, 0.5))
}

chordMF <- function(k, a = 0.99) {
  sqrt(2 * (1 - t_func(k, a))) / sqrt(2 * (1 - t_func(k, 0.5)))
}

# Defining the function that calculates the multiplying factor given the set of points
# points should be a matrix with rows as points, dist should be one of "arc", "cos" or "chord",
# a is the probability mass that you would like to lie in the loop (given the reference distribution)
MF <- function(points, a = 0.99, dist = "arc") {
  coords <- points
  pointdepths <- ahD(coords, rep(1, nrow(coords))) / nrow(coords)
  med <- median(pointdepths)
  bagdepth <- pointdepths[pointdepths >= med]
  bagpoints <- points[pointdepths >= med, , drop = FALSE]
  # Assuming mid is the point with maximum depth (spherical median)
  mid <- points[which.max(pointdepths), ]
  
  if (dist == "arc") {
    maxd <- max(apply(bagpoints, 1, function(p) arcdist(p, mid)))
    return(arcMF(kappaarc(maxd), a))
  } else if (dist == "chord") {
    maxd <- max(apply(bagpoints, 1, function(p) chorddist(p, mid)))
    return(chordMF(kappachord(maxd), a))
  } else if (dist == "cos") {
    maxd <- max(apply(bagpoints, 1, function(p) cosdist(p, mid)))
    return(cosMF(kappacos(maxd), a))
  } else {
    stop("Wrong format of dist")
  }
}

# Note: The plotting utilities (generate_sphere_data, ortho_matrix, combine_images) are Python-specific
# and would need to be adapted to R using packages like rgl, plotly, or magick for visualization.

# ortho_matrix function
ortho_matrix <- function(v1) {
  v1 <- as.numeric(v1)
  
  # Find a temporary vector not collinear with v1
  standard_bases <- matrix(c(1, 0, 0, 0, 1, 0, 0, 0, 1), nrow = 3, byrow = TRUE)
  
  dot_products <- abs(standard_bases %*% v1)
  
  temp_vector_index <- which.min(dot_products)
  temp_vector <- standard_bases[temp_vector_index, ]
  
  v2_prime <- c(
    v1[2] * temp_vector[3] - v1[3] * temp_vector[2],
    v1[3] * temp_vector[1] - v1[1] * temp_vector[3],
    v1[1] * temp_vector[2] - v1[2] * temp_vector[1]
  )
  
  norm_v2_prime <- sqrt(sum(v2_prime^2))
  if (norm_v2_prime == 0) {
    stop("Could not find a non-zero cross product for v2.")
  }
  
  u2 <- v2_prime / norm_v2_prime
  u3 <- c(
    v1[2] * u2[3] - v1[3] * u2[2],
    v1[3] * u2[1] - v1[1] * u2[3],
    v1[1] * u2[2] - v1[2] * u2[1]
  )
  
  return(t(matrix(c(u2, u3, v1), nrow = 3)))
}
