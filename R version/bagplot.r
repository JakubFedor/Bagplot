# Install and load required packages
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools", repos = "http://cran.us.r-project.org")
}
devtools::install_github("DyckerhoffRainer/sphericalDepth")
library(sphericalDepth)
library(plotly)  # For plotting, assuming plotly for R is installed
library(rgl)     # Alternative 3D plotting
library(sf)      # For geographic data
library(rnaturalearth)  # For continent outlines

# Source the fun.r file (assuming it's in the same directory)
source("fun.r")

# Function to plot continent outlines on the sphere
plot_continent_outlines_on_sphere <- function(fig, c = NULL, r = NULL) {
  # Get world countries data
  world <- ne_countries(scale = "medium", returnclass = "sf")
  
  # Iterate through each country
  for (i in 1:nrow(world)) {
    geom <- st_geometry(world[i, ])
    
    # Handle polygons
    if (inherits(geom[[1]], "POLYGON") || inherits(geom[[1]], "MULTIPOLYGON")) {
      coords <- st_coordinates(geom)
      lons <- coords[, 1]
      lats <- coords[, 2]
      
      # Convert to Cartesian
      lat_rad <- lats * pi / 180
      lon_rad <- lons * pi / 180
      x <- cos(lat_rad) * cos(lon_rad)
      y <- cos(lat_rad) * sin(lon_rad)
      z <- sin(lat_rad)
      
      # Add to plotly figure (assuming fig is a plotly object)
      if (is.null(c) || is.null(r)) {
        fig <- fig %>% add_trace(
          type = "scatter3d",
          x = x, y = y, z = z,
          mode = "lines",
          line = list(color = "black", width = 1),
          showlegend = FALSE
        )
      } else {
        # For subplots, this would need adjustment
        fig <- fig %>% add_trace(
          type = "scatter3d",
          x = x, y = y, z = z,
          mode = "lines",
          line = list(color = "black", width = 1),
          showlegend = FALSE,
          xaxis = paste0("x", if(!is.null(c)) c else "", if(!is.null(r)) r else ""),
          yaxis = paste0("y", if(!is.null(c)) c else "", if(!is.null(r)) r else "")
        )
      }
    }
  }
  return(fig)
}

# Main bagplot function
bagplot <- function(data, weights = NULL, dist = "arc", a = 0.99, borderdist = "mean", res = 500,
                    savefig = FALSE, figname = "bagplot.pdf", interactive = TRUE,
                    bagcol = "#3c7cdd", loopcol = "#a8c5f0", font = "Latin Modern Roman", geo = FALSE) {
  
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
  
  # Find border depth
  borderdepth <- median(datadepth)
  
  # Find points with max depth
  max_depth_indices <- which(datadepth == max(datadepth))
  if (length(max_depth_indices) == 1) {
    mid <- data[max_depth_indices, ]
  } else {
    midmean <- colMeans(data[max_depth_indices, , drop = FALSE])
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
  rotation_matrix <- ortho_matrix(mid)  # Assuming ortho_matrix is defined in fun.r
  
  # Apply rotation
  x_rot <- rotation_matrix[1,1] * x_sphere + rotation_matrix[1,2] * y_sphere + rotation_matrix[1,3] * z_sphere
  y_rot <- rotation_matrix[2,1] * x_sphere + rotation_matrix[2,2] * y_sphere + rotation_matrix[2,3] * z_sphere
  z_rot <- rotation_matrix[3,1] * x_sphere + rotation_matrix[3,2] * y_sphere + rotation_matrix[3,3] * z_sphere
  
  # Color the surface
  surface_colors <- matrix(0, nrow = res, ncol = res)
  for (i in 1:nrow(grid)) {
    point <- c(x_rot[i], y_rot[i], z_rot[i])
    depth <- ahD(data, weights, matrix(point, nrow = 1))
    surface_colors[i] <- ifelse(depth > borderdepth, 1, 0)
  }
  
  # Find borders and distances
  borders <- list()
  for (j in 1:ncol(surface_colors)) {
    zero_indices <- which(surface_colors[, j] == 0)
    if (length(zero_indices) > 0) {
      borders[[j]] <- c(zero_indices[1] - 1, j)
    }
  }
  
  middist <- matrix(0, nrow = res, ncol = res)
  for (i in 1:nrow(grid)) {
    point <- c(x_rot[i], y_rot[i], z_rot[i])
    if (dist == "arc") {
      middist[i] <- arcdist(point, mid)
    } else if (dist == "cos") {
      middist[i] <- cosdist(point, mid)
    } else {
      middist[i] <- chorddist(point, mid)
    }
  }
  
  # Calculate border distance
  border_distances <- sapply(borders, function(b) middist[b[1], b[2]])
  if (borderdist == "max") {
    borderd <- max(border_distances)
  } else {
    borderd <- mean(border_distances)
  }
  
  # Calculate factor
  if (dist == "arc") {
    factor <- arcMF(kappaarc(borderd), a)
  } else if (dist == "cos") {
    factor <- cosMF(kappacos(borderd), a)
  } else {
    factor <- chordMF(kappachord(borderd), a)
  }
  
  # Update surface colors for loop
  for (j in 1:ncol(surface_colors)) {
    borderd_j <- middist[borders[[j]][1], borders[[j]][2]]
    loopd <- factor * borderd_j
    for (i in 1:nrow(surface_colors)) {
      if (middist[i, j] > borderd_j && middist[i, j] <= loopd) {
        surface_colors[i, j] <- 0.5
      }
    }
  }
  
  # Plotting (simplified, using plotly)
  custom_colorscale <- list(
    list(0, 'lightgrey'),
    list(0.5, loopcol),
    list(1, bagcol)
  )
  
  if (interactive) {
    fig <- plot_ly() %>%
      add_surface(
        x = matrix(x_rot, nrow = res), 
        y = matrix(y_rot, nrow = res), 
        z = matrix(z_rot, nrow = res),
        surfacecolor = surface_colors,
        colorscale = custom_colorscale,
        cmin = 0, cmax = 1,
        showscale = FALSE
      )
    
    # Add data points
    for (i in 1:nrow(data)) {
      fig <- fig %>% add_trace(
        type = "scatter3d",
        x = data_x[i], y = data_y[i], z = data_z[i],
        mode = "markers",
        marker = list(size = 5, color = "black")
      )
    }
    
    # Add center
    fig <- fig %>% add_trace(
      type = "scatter3d",
      x = mid[1], y = mid[2], z = mid[3],
      mode = "markers",
      marker = list(size = 10, color = "black")
    )
    
    if (geo) {
      fig <- plot_continent_outlines_on_sphere(fig)
    }
    
    fig <- fig %>% layout(
      title = "Bagplot",
      scene = list(
        xaxis = list(visible = FALSE),
        yaxis = list(visible = FALSE),
        zaxis = list(visible = FALSE)
      )
    )
    
    print(fig)  # In R, use print to display plotly
  } else {
    # For non-interactive, this would need more work for subplots and saving
    # Using rgl as alternative
    rgl::open3d()
    rgl::surface3d(x_rot, y_rot, z_rot, color = surface_colors, col = custom_colorscale)
    # Add points and center similarly
    rgl::points3d(data, size = 5)
    rgl::points3d(mid, size = 10)
    if (geo) {
      # Add continent outlines using rgl
      # This would require converting the outlines to 3D lines
    }
    rgl::rgl.snapshot(figname)  # Save if needed
  }
}

# Note: The ortho_matrix function needs to be defined in fun.r
# The plotting is adapted to use plotly or rgl, but may need further refinement for exact replication.
