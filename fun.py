import rpy2.robjects as robjects
import pandas as pd
from rpy2.robjects.packages import importr
from numpy.ma.core import arccos
from rpy2.robjects.vectors import FloatVector, FloatMatrix
import plotly.graph_objects as go
import numpy as np
import scipy
import math
from PIL import Image
import os
import shutil # For removing directories and their contents
base = importr('base')
#Importing packages and making sure that the correct R packages are imported and installed as well.

utils = importr('utils')
devtools=importr('devtools')
# robjects.r('''
#   if (!requireNamespace("devtools", quietly = TRUE)) {
#     install.packages("devtools", repos = "http://cran.us.r-project.org")
#   }
#   devtools::install_github("DyckerhoffRainer/sphericalDepth")  # Example: installing ggplot2 from GitHub
# ''')
ahd=importr('sphericalDepth')

#Defining function t, this will be used later to estimate the parameter kappa and
def t(k,a=0.99):
    if k!=0:
        return (math.log((1-a)*math.exp(2*k)+a)-k)/k
    if k==0:
        return 1-2*a


#Definitions of distance metrics.
def arcdist(x,y):
    return np.arccos(np.dot(x,y))
def cosdist(x,y):
    return 1-np.dot(x,y)
def chorddist(x,y):
    return np.sqrt(2*(1-np.dot(x,y)))


#Definition of the eqautions to be solved to estimate kappa.
def arceq(k,d):
    return arccos(t(k,a=0.5))-d
def coseq(k,d):
    return 1-t(k,a=0.5)-d
def chordeq(k,d):
    return np.sqrt(2*(1-t(k,a=0.5)))-d
#Defining functions for the estimation for kappa.
def kappaarc(d):
    return scipy.optimize.fsolve(arceq,1,args=(d))
def kappacos(d):
    return scipy.optimize.fsolve(coseq,1,args=(d))
def kappachord(d):
    return scipy.optimize.fsolve(chordeq,1,args=(d))

#Defining functions that calculate the multiplying factor, given the estimate of kappa
def arcMF(k,a=0.99):
    return np.arccos(t(k,a))/np.arccos(t(k,1/2))[0]
def cosMF(k,a=0.99):
    return (1-t(k,a))/(1-t(k,1/2))[0]
def chordMF(k,a=0.99):
    return np.sqrt(2*(1-t(k,a)))/np.sqrt(2*(1-t(k,1/2)))[0]


#defining the function that calculates the multiplying factor given the set of points
#points should be np.ndarray, dist should be one of "arc", "cos" or "chord", a is the probability mass that you would
# like to lie in the loop (given the reference distribution)

def MF(points, a=0.99, dist="arc"):
    x=FloatVector(points[:,0])
    y=FloatVector(points[:,1])
    z=FloatVector(points[:,2])
    robjects.globalenv['x'] = x
    robjects.globalenv['y'] = y
    robjects.globalenv['z'] = z
    robjects.r("coords=cbind(x,y,z)")
    pointdepths=robjects.r("ahD(coords,rep(1,length(x)))/length(x)")
    pointdepths=np.asarray(pointdepths)
    dictdepths=dict(zip(pointdepths, points))
    med=np.median(pointdepths)
    bagdepth=[depth for depth in pointdepths if depth>=med]
    bagpoints=pd.DataFrame([dictdepths[key] for key in bagdepth]).to_numpy()
    if dist == "arc":
        maxd=max([arcdist(point,mid) for point in bagpoints])
        return arcMF(kappaarc(maxd),a)[0]
    elif dist == "chord":
        maxd=max([chorddist(point,mid) for point in bagpoints])
        return chordMF(kappachord(maxd),a)[0]
    elif dist == "cos":
        maxd=max([cosdist(point,mid) for point in bagpoints])
        return cosMF(kappacos(maxd),a)[0]
    else:
        print("Wrong format of dist")
        return


#function used when plotting the data points for a better visibility (we create small spheres instead of standard
# 3d scatter plots that use 2d objects.
def generate_sphere_data(center, radius, segments_theta=20, segments_phi=20):
    """
    Generates vertices and faces for a sphere.

    Args:
        center (list or tuple): [x, y, z] coordinates of the sphere's center.
        radius (float): The radius of the sphere.
        segments_theta (int): Number of segments along the theta (longitudinal) axis.
        segments_phi (int): Number of segments along the phi (latitudinal) axis.

    Returns:
        tuple: (x_flat, y_flat, z_flat, i_faces, j_faces, k_faces)
               - x_flat, y_flat, z_flat: Flattened arrays of vertex coordinates.
               - i_faces, j_faces, k_faces: Lists of vertex indices forming triangles.
    """
    cx, cy, cz = center

    # Create a grid of spherical coordinates
    # theta: longitude (0 to 2*pi)
    # phi: latitude (0 to pi)
    theta = np.linspace(0, 2 * np.pi, segments_theta, endpoint=False) # Exclude endpoint to avoid duplicate vertices
    phi = np.linspace(0, np.pi, segments_phi)

    # Convert spherical to Cartesian coordinates for a unit sphere
    x_unit = np.outer(np.cos(theta), np.sin(phi))
    y_unit = np.outer(np.sin(theta), np.sin(phi))
    z_unit = np.outer(np.ones(np.size(theta)), np.cos(phi))

    # Scale and translate to the desired radius and center
    x = cx + radius * x_unit
    y = cy + radius * y_unit
    z = cz + radius * z_unit

    # Flatten the arrays of vertices for Mesh3d
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Generate faces (triangles) from the grid
    # Each "quad" (4 points) from the grid forms two triangles.
    i_faces = []
    j_faces = []
    k_faces = []

    rows, cols = x.shape # Shape of the grid (segments_theta, segments_phi)

    # Iterate through all 'quads' formed by adjacent grid points
    for r in range(rows):
        for c in range(cols - 1):
            # Vertices of the current quad, handling wrap-around for theta
            v0 = r * cols + c
            v1 = r * cols + (c + 1)
            v2 = ((r + 1) % rows) * cols + (c + 1) # Use modulo for wrapping theta
            v3 = ((r + 1) % rows) * cols + c

            # First triangle
            i_faces.append(v0)
            j_faces.append(v1)
            k_faces.append(v2)

            # Second triangle
            i_faces.append(v0)
            j_faces.append(v2)
            k_faces.append(v3)
    return x_flat, y_flat, z_flat, i_faces, j_faces, k_faces


#find an orthogonal matrix that we use to transform the grid in order to guarantee
def ortho_matrix(v1):
    """
    Finds two more vectors to form an orthonormal basis in R^3
    with the given vector v1.

    Args:
        v1 (np.array): A 3D numpy array representing the initial vector.

    Returns:
        A matrix needed for a tranformation of a 3D point to a new coordinate system.
    """
    v1 = np.array(v1, dtype=float)

    # 2. Find a temporary vector not collinear with u1
    # We choose the standard basis vector that has the smallest absolute dot product with u1.
    # This minimizes the chance of their cross product being zero or very small.
    standard_bases = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Calculate absolute dot products with u1
    dot_products = np.abs(np.dot(standard_bases, v1))

    # Get the index of the standard basis vector that is "most orthogonal" to u1
    # (i.e., has the smallest absolute dot product)
    temp_vector_index = np.argmin(dot_products)
    temp_vector = standard_bases[temp_vector_index]

    # 3. Calculate the second orthogonal vector (u2)
    v2_prime = np.cross(v1, temp_vector)


    norm_v2_prime = np.linalg.norm(v2_prime)
    if norm_v2_prime == 0:
        print("Error: Could not find a non-zero cross product for u2. This is highly unexpected.")
        return None, None, None

    u2 = v2_prime / norm_v2_prime

    # 4. Calculate the third orthogonal vector (u3)
    u3 = np.cross(v1, u2)

    return np.transpose([u2, u3, v1])

#function used to combine images, used in the plotting of the images when savefig=True and  interactive=False
def combine_images(image_filenames, output_filename, rows, cols, delete_source_images=True):
    """
    Combines a list of image files into a single composite image.

    Args:
        image_filenames (list): A list of paths to the individual image files.
                                 Images are expected to be ordered left-to-right, then top-to-bottom.
        output_filename (str): The path and name for the combined output image (e.g., 'combined_plot.png').
        rows (int): The number of rows in the grid for combining images.
        cols (int): The number of columns in the grid for combining images.
        delete_source_images (bool): If True, deletes the individual image files
                                     after they have been successfully combined.
    """
    if not image_filenames:
        print("No image filenames provided. Nothing to combine.")
        return

    images = []
    # Store successfully loaded image paths for potential deletion
    loaded_image_paths = []

    for filename in image_filenames:
        try:
            img = Image.open(filename)
            images.append(img)
            loaded_image_paths.append(filename) # Only add if successfully opened
        except FileNotFoundError:
            print(f"Warning: Image file not found: {filename}. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Could not open image {filename}: {e}. Skipping.")
            continue

    if not images:
        print("No valid images could be loaded. Cannot combine.")
        return

    # Get dimensions from the first image (assuming all images are the same size)
    img_width, img_height = images[0].size

    total_width = img_width * cols
    total_height = img_height * rows

    combined_img = Image.new('RGB', (total_width, total_height), color='white') # White background

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            print(f"Warning: More images provided ({len(images)}) than grid cells ({rows*cols}). Extra images will be ignored.")
            break

        row_idx = idx // cols
        col_idx = idx % cols

        x_offset = col_idx * img_width
        y_offset = row_idx * img_height

        combined_img.paste(img, (x_offset, y_offset))

    combined_img.save(output_filename)
    print(f"Image saved as '{output_filename}'")

    # --- Deletion Logic ---
    if delete_source_images:
        for img_path in loaded_image_paths:
            try:
                os.remove(img_path)
            except OSError as e:
                print(f"Error deleting file {img_path}: {e}")
