import fun
import rpy2.robjects as robjects
import pandas as pd
from rpy2.robjects.packages import importr
from numpy.ma.core import arccos
from rpy2.robjects.vectors import FloatVector, FloatMatrix
import plotly.graph_objects as go
import numpy as np
import scipy
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import os
import shutil
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
import requests


#Importing packages and making sure that the correct R packages are imported and installed as well.

base = importr('base')
# import R's "utils" package
utils = importr('utils')
devtools=importr('devtools')
# robjects.r('''
#   if (!requireNamespace("devtools", quietly = TRUE)) {
#     install.packages("devtools", repos = "http://cran.us.r-project.org")
#   }
#   devtools::install_github("DyckerhoffRainer/sphericalDepth")  # Example: installing ggplot2 from GitHub
# ''')
ahd=importr('sphericalDepth')


def plot_continent_outlines_on_sphere(fig,c=None,r=None):
    """
    Plots continent outlines on an existing Plotly 3D figure representing a unit sphere.

    Args:
        fig (go.Figure): The existing Plotly figure to which the outlines will be added.
    """

    # 1. Acquire GeoJSON data for country boundaries (which implicitly define continent outlines)
    url = "https://raw.githubusercontent.com/datasets/geo-countries/main/data/countries.geojson"
    response = requests.get(url)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    geojson_data = response.json()
    world = gpd.GeoDataFrame.from_features(geojson_data["features"])


    # 2. Iterate through each country's geometry to extract boundary coordinates
    for _, row in world.iterrows():
        geometry = row['geometry']

        # Handle both single polygons and multi-part polygons (like islands)
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = geometry.geoms
        else:
            continue # Skip other geometry types (e.g., points, lines if they were present)

        for poly in polygons:
            exterior_coords = np.array(poly.exterior.coords)

            lons = exterior_coords[:, 0]
            lats = exterior_coords[:, 1]

            # 3. Convert geographic (lat, lon) to Cartesian (x, y, z) for a unit sphere
            lat_rad = np.deg2rad(lats)
            lon_rad = np.deg2rad(lons)

            x = np.cos(lat_rad) * np.cos(lon_rad)
            y = np.cos(lat_rad) * np.sin(lon_rad)
            z = np.sin(lat_rad)

            # 4. Add the generated outline as a 3D line trace to the Plotly figure
            #    A new trace is created for each country's outline.
            if c==None or r==None:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False))

            else:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False),
                row=r,col=c)

    return fig





#The function that plots the bagplot.

def bagplot(data, dist="arc", a=0.99 ,borderdist="mean", res=500, savefig=False, figname="bagplot.pdf",  interactive=True, bagcol="#3c7cdd", loopcol="#a8c5f0",font="Latin Modern Roman",geo=False):
    """
        plots the bagplot as described in the thesis

        Args:
            data (np.ndarray or data format that can be transformed into an np.ndarray): Datapoints for wich the bagplot is to be plotted.
            dist ("arc", "cos" or "chord"): The distance to be used in calculations of the estimate of kappa and multiplying factor for the loop.
            borderdist ("max", "mean"): Whether the maximal or mean distance from the center of the bagplot to the border of the bag should be used.
            res (integer): Higher number will result in a better aproximation of the sphere.
            savefig (bool): whether or not the resulting figure should be saved, if True interactive and interactive is true as well, the figure will not be saved.
            interactive (bool): Whether or not the resulting plot should be shown interactively. For True, one interative sphere will be plotted.
            For False, 6 subplots will be plotted, showing the sphere from above, below, front, back, right and left.
            res (integer): Higher number will result in a higher resolution of the sphere.
            bagcol (str): Color of the bag.
            loopcol (str): Color of the loop.
            font (str): Font used in the plot.

        Returns:
            plots the spherical bagplot
        """
    data=np.asarray(data)
    #x, y and z values from the data
    data_x=data[:, 0]
    data_y=data[:, 1]
    data_z=data[:, 2]
    #loading the data into R instance
    fv_x = FloatVector(data_x)
    fv_y = FloatVector(data_y)
    fv_z = FloatVector(data_z)
    robjects.globalenv['x'] = fv_x
    robjects.globalenv['y'] = fv_y
    robjects.globalenv['z'] = fv_z
    robjects.r("data=cbind(x,y,z)")
    robjects.r("l=length(x)")
    #calculate the depth of the datapoints
    datadepth=robjects.r("ahD(data,rep(1,l))/l")
    #finding the depth of the border of the bag
    borderdepth = np.median(datadepth)
    datadict = dict()
    for key in datadepth:
        datadict[key] = []
        indices = [i for i, x in enumerate(datadepth) if x == key]
        for i in indices:
            datadict[key].append(data[i])
    if len(datadict[max(datadepth)]) == 1:
        mid = datadict[max(datadepth)][0]
    else:
        midmean = np.mean(datadict[max(datadepth)], axis=0)
        mid = midmean / np.linalg.norm(midmean)
    #generating the grid for plotting the sphere
    phi, theta = np.mgrid[0:np.pi:res*1j, 0:2 * np.pi:res*1j]
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)



    #fingind the rotation matrix that ensures that the first point in the grid used to generate the sphere
    rotation_matrix = fun.ortho_matrix(mid)

    x_sphere_rotated = (rotation_matrix[0, 0] * x_sphere +
                        rotation_matrix[0, 1] * y_sphere +
                        rotation_matrix[0, 2] * z_sphere)

    y_sphere_rotated = (rotation_matrix[1, 0] * x_sphere +
                        rotation_matrix[1, 1] * y_sphere +
                        rotation_matrix[1, 2] * z_sphere)

    z_sphere_rotated = (rotation_matrix[2, 0] * x_sphere +
                        rotation_matrix[2, 1] * y_sphere +
                        rotation_matrix[2, 2] * z_sphere)

    custom_colorscale = [
        [0, 'lightgrey'],
        [0.5, loopcol],
        [1, bagcol]
    ]
    #generating the grid used to color the sphere
    surface_colors_data = x_sphere_rotated.copy()
    for i in range(len(x_sphere)):
        x_spherefv = FloatVector(x_sphere_rotated[i])
        y_spherefv = FloatVector(y_sphere_rotated[i])
        z_spherefv = FloatVector(z_sphere_rotated[i])
        robjects.globalenv['x'] = x_spherefv
        robjects.globalenv['y'] = y_spherefv
        robjects.globalenv['z'] = z_spherefv
        robjects.r("coords=cbind(x,y,z)")
        surface_colors_data[i] = np.where(np.array(robjects.r("ahD(data,rep(1,l),coords)/l")) >= borderdepth,
                                          1, 0)
    #finding the borders of the bag
    borders = []
    for i in range(len(x_sphere_rotated)):
        borders.append([np.where(surface_colors_data[:, i] == 0)[0][0] - 1, i])
    middist = x_sphere_rotated.copy()
    for i in range(len(x_sphere_rotated)):
        for j in range(len(x_sphere_rotated[i])):
            if dist=="arc":
                middist[i, j] = fun.arcdist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]], mid)
            elif dist=="cos":
                middist[i,j]= fun.cosdist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]], mid)
            else:
                middist[i, j] = fun.chorddist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]],mid)
    #finding the maximal distance from the center of the bagplot to the border of the bag
    if borderdist == "max":
        borderd = max([middist[b[0], b[1]] for b in borders])
    if borderdist == "mean":
        borderd = np.mean([middist[b[0], b[1]] for b in borders])
    if dist == "arc":
        factor = fun.arcMF(fun.kappaarc(borderd),a)
    elif dist == "cos":
        factor = fun.cosMF(fun.kappacos(borderd),a)
    elif dist == "chord":
        factor = fun.chordMF(fun.kappachord(borderd),a)
    else:
        print("Wrong format of dist")
        return

    #changing the grid used to color the sphere (now in addition to bag we can also color the loop)
    for j in range(len(surface_colors_data)):
        borderd=middist[borders[j][0],borders[j][1]]
        loopd=factor*borderd
        for i in range(len(surface_colors_data[j])):
            if middist[i,j]> borderd and middist[i,j] <=loopd:
                surface_colors_data[i,j]=0.5
    #the plotting itself depending on the choice of interactive and savefig



    if not geo:
        if interactive:
            if savefig:
                print("Saving figure is not adviced with interactive mode. Proceeding without saving.")

            fig = go.Figure(
                data=[
                    go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                               surfacecolor=surface_colors_data,
                               colorscale=custom_colorscale,
                               cmin=0, cmax=1,
                               opacity=1,
                               showscale=False,
                               name='Sphere Regions')]
            )

            for i in range(len(data_x)):
                center = [data_x[i], data_y[i], data_z[i]]
                # Generate sphere mesh data for the current center
                x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                    center, 0.005, 15, 15
                )


                fig.add_trace(go.Mesh3d(
                    x=x_sphere1, y=y_sphere1, z=z_sphere1,
                    i=i_faces, j=j_faces, k=k_faces,
                    color="black",
                    opacity=1,
                    showlegend=False
                ))



            center = [mid[0], mid[1], mid[2]]
            # Generate sphere mesh data for the current center
            x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                center, 0.05, 15, 15
            )


            fig.add_trace(go.Mesh3d(
                x=x_sphere1, y=y_sphere1, z=z_sphere1,
                i=i_faces, j=j_faces, k=k_faces,
                color="black",
                opacity=1,
                showlegend=False
            ))
            # Update layout
            fig.update_layout(
                title='Bagplot',
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),

                    bgcolor="rgba(0,0,0,0)",
                    aspectmode='data',
                    camera={
                        "center": dict(x=0, y=0, z=0),  # Look at origin
                        "up": dict(x=0, y=0, z=1)
                    }
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                font=dict(
                    family=font,
                    size=25)
            )
            fig.show()
        else:
            if not savefig:


                fig = make_subplots(
                    rows=2, cols=3,
                    specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
                           [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=list(
                        ['View from above', 'View from below', 'View from the front', 'View from the back', 'View from the right',
                         'View from the left'])
                )

                row = 1
                col = 1
                for i in range(6):
                    # Add the sphere surface
                    fig.add_trace(
                        go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                                   surfacecolor=surface_colors_data,  # <--- Pass your conditional data here
                                   colorscale=custom_colorscale,  # <--- Apply your custom colorscale
                                   cmin=0, cmax=1,  # <--- Define the range of values in surface_colors_data
                                   opacity=1,
                                   showscale=False ),
                        row=row, col=col
                    )


                    # Add the random points (the 'dots')
                    for i in range(len(data_x)):
                        center = [data_x[i], data_y[i], data_z[i]]
                        # Generate sphere mesh data for the current center
                        x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                            center, 0.005, 15, 15
                        )

                        # Add a Mesh3d trace for this sphere
                        fig.add_trace(go.Mesh3d(
                            x=x_sphere1, y=y_sphere1, z=z_sphere1,
                            i=i_faces, j=j_faces, k=k_faces,
                            color="black",  # Assign a random color for each sphere
                            opacity=1,
                            showlegend=False
                        ),
                            row=row, col=col)


                    center = [mid[0], mid[1], mid[2]]
                    # Generate sphere mesh data for the current center
                    x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                        center, 0.05, 15, 15
                    )

                    # Add a Mesh3d trace for this sphere
                    fig.add_trace(go.Mesh3d(
                        x=x_sphere1, y=y_sphere1, z=z_sphere1,
                        i=i_faces, j=j_faces, k=k_faces,
                        color="black",  # Assign a random color for each sphere
                        opacity=1,
                        showlegend=False
                    ),
                        row=row, col=col)
                    # Update scene layout for each subplot

                    col += 1
                    if col > 3:
                        col = 1
                        row += 1

                fig.update_layout(
                    title_text="Bagplot from different views",
                    height=800,  # Adjust height as needed
                    width=1000,  # Adjust width as needed
                    margin=dict(l=50, r=50, b=50, t=100),
                    showlegend=False,
                    plot_bgcolor='white',  # Background of the entire plot area
                    paper_bgcolor='white',  # Background outside the plot area
                    scene_camera={"eye": {'x': .0, 'y': 0, 'z': 1.25}},
                    scene_xaxis_visible=False,
                    scene_yaxis_visible=False,
                    scene_zaxis_visible=False,
                    scene_aspectmode="cube",
                    scene_bgcolor="rgba(0,0,0,0)",
                    scene2_camera={"eye": {'x': .0, 'y': 0, 'z': -1.25}},
                    scene2_xaxis_visible=False,
                    scene2_yaxis_visible=False,
                    scene2_zaxis_visible=False,
                    scene2_aspectmode="cube",
                    scene2_bgcolor="rgba(0,0,0,0)",
                    scene3_camera={"eye": {'x': .0, 'y': 1.25, 'z': 0}},
                    scene3_xaxis_visible=False,
                    scene3_yaxis_visible=False,
                    scene3_zaxis_visible=False,
                    scene3_aspectmode="cube",
                    scene3_bgcolor="rgba(0,0,0,0)",
                    scene4_camera={"eye": {'x': 0, 'y': -1.25, 'z': 0}},
                    scene4_xaxis_visible=False,
                    scene4_yaxis_visible=False,
                    scene4_zaxis_visible=False,
                    scene4_aspectmode="cube",
                    scene4_bgcolor="rgba(0,0,0,0)",
                    scene5_camera={"eye": {'x': 1.25, 'y': 0, 'z': 0}},
                    scene5_xaxis_visible=False,
                    scene5_yaxis_visible=False,
                    scene5_zaxis_visible=False,
                    scene5_aspectmode="cube",
                    scene5_bgcolor="rgba(0,0,0,0)",
                    scene6_camera={"eye": {'x': -1.25, 'y': 0, 'z': 0}},
                    scene6_xaxis_visible=False,
                    scene6_yaxis_visible=False,
                    scene6_zaxis_visible=False,
                    scene6_aspectmode="cube",
                    scene6_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family=font,
                        size=25)

                )

                fig.show()


            else:
                titles = ['View from above', 'View from below', 'View from the front', 'View from the back',
                          'View from the right', 'View from the left']
                image_names = ["bagabove.png", "bagbelow.png", "bagfront.png", "bagback.png", "bagright.png", "bagleft.png"]
                cameras = [{"eye": {'x': .0, 'y': 0, 'z': 1.25}}, {"eye": {'x': .0, 'y': 0, 'z': -1.25}},
                           {"eye": {'x': .0, 'y': 1.25, 'z': 0}}, {"eye": {'x': .0, 'y': -1.25, 'z': 0}},
                           {"eye": {'x': 1.25, 'y': 0, 'z': 0}}, {"eye": {'x': -1.25, 'y': 0, 'z': 0}}]

                for a in range(6):
                    fig = go.Figure(
                        data=[
                            go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                                       surfacecolor=surface_colors_data,  # <--- Pass your conditional data here
                                       colorscale=custom_colorscale,  # <--- Apply your custom colorscale
                                       cmin=0, cmax=1,  # <--- Define the range of values in surface_colors_data
                                       opacity=1,
                                       showscale=False,  # Hide the color scale bar
                                       name='Sphere Regions')]
                    )


                    for i in range(len(data_x)):
                        center = [data_x[i], data_y[i], data_z[i]]
                        # Generate sphere mesh data for the current center
                        x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                            center, 0.005, 15, 15
                        )

                        # Add a Mesh3d trace for this sphere
                        fig.add_trace(go.Mesh3d(
                            x=x_sphere1, y=y_sphere1, z=z_sphere1,
                            i=i_faces, j=j_faces, k=k_faces,
                            color="black",  # Assign a random color for each sphere
                            opacity=1,
                            showlegend=False
                        ))


                    center = [mid[0], mid[1], mid[2]]
                    # Generate sphere mesh data for the current center
                    x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                        center, 0.05, 15, 15
                    )

                    # Add a Mesh3d trace for this sphere
                    fig.add_trace(go.Mesh3d(
                        x=x_sphere1, y=y_sphere1, z=z_sphere1,
                        i=i_faces, j=j_faces, k=k_faces,
                        color="black",  # Assign a random color for each sphere
                        opacity=1,
                        showlegend=False
                    ))
                    # Update layout
                    fig.update_layout(
                        title=titles[a],
                        title_x=0.5,
                        scene=dict(
                            xaxis_visible=False,
                            yaxis_visible=False,
                            zaxis_visible=False,
                            bgcolor="rgba(0,0,0,0)",
                            aspectmode="cube",
                            camera=cameras[a]
                        ),
                        margin=dict(l=0, r=0, b=0, t=50),
                        font=dict(
                            family=font,
                            size=25)
                    )

                    fig.write_image(image_names[a], engine="kaleido")

                fun.combine_images(image_names, figname, 2, 3)

    if geo:
        contcenters_xyz=[[0.5197807100754235, 0.24588422745439137, 0.8181497174250234], \
                         [0.033674887449759994, 0.7224104283095965, 0.6906440291675526],\
                         [0.9597773801926751, 0.277025506104953, 0.045653580777193926],\
                         [-0.12742624951896128, -0.6617151315913206, 0.7388475049403719],\
                         [0.5372527575923208, -0.7994243397696936, -0.2688497711422428],\
                         [-0.6304174044492995, 0.6447776767078117, -0.432244888676182]]
        if interactive:
            if savefig:
                print("Saving figure is not adviced with interactive mode. Proceeding without saving.")

            fig = go.Figure(
                data=[
                    go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                               surfacecolor=surface_colors_data,
                               colorscale=custom_colorscale,
                               cmin=0, cmax=1,
                               opacity=1,
                               showscale=False,
                               name='Sphere Regions')]
            )

            for i in range(len(data_x)):
                center = [data_x[i], data_y[i], data_z[i]]
                # Generate sphere mesh data for the current center
                x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                    center, 0.005, 15, 15
                )


                fig.add_trace(go.Mesh3d(
                    x=x_sphere1, y=y_sphere1, z=z_sphere1,
                    i=i_faces, j=j_faces, k=k_faces,
                    color="black",
                    opacity=1,
                    showlegend=False
                ))



            center = [mid[0], mid[1], mid[2]]
            # Generate sphere mesh data for the current center
            x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                center, 0.05, 15, 15
            )


            fig.add_trace(go.Mesh3d(
                x=x_sphere1, y=y_sphere1, z=z_sphere1,
                i=i_faces, j=j_faces, k=k_faces,
                color="black",
                opacity=1,
                showlegend=False
            ))
            # Update layout
            fig.update_layout(
                title='Bagplot',
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),

                    bgcolor="rgba(0,0,0,0)",
                    aspectmode='data',
                    camera={
                        "center": dict(x=0, y=0, z=0),  # Look at origin
                        "up": dict(x=0, y=0, z=1)
                    }
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                font=dict(
                    family=font,
                    size=25)
            )
            fig=plot_continent_outlines_on_sphere(fig)
            fig.show()
        else:
            if not savefig:


                fig = make_subplots(
                    rows=2, cols=3,
                    specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
                           [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=list(
                        ['Europe', 'Asia', "Africa", 'North America', 'South America', 'Australia'
                         ])
                )

                row = 1
                col = 1
                for i in range(6):
                    # Add the sphere surface
                    fig.add_trace(
                        go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                                   surfacecolor=surface_colors_data,
                                   colorscale=custom_colorscale,
                                   cmin=0, cmax=1,
                                   opacity=1,
                                   showscale=False ),
                        row=row, col=col
                    )


                    # Add the random points (the 'dots')
                    for i in range(len(data_x)):
                        center = [data_x[i], data_y[i], data_z[i]]
                        # Generate sphere mesh data for the current center
                        x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                            center, 0.005, 15, 15
                        )

                        # Add a Mesh3d trace for this sphere
                        fig.add_trace(go.Mesh3d(
                            x=x_sphere1, y=y_sphere1, z=z_sphere1,
                            i=i_faces, j=j_faces, k=k_faces,
                            color="black",  # Assign a random color for each sphere
                            opacity=1,
                            showlegend=False
                        ),
                            row=row, col=col)


                    center = [mid[0], mid[1], mid[2]]
                    # Generate sphere mesh data for the current center
                    x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                        center, 0.05, 15, 15
                    )


                    # Add a Mesh3d trace for this sphere
                    fig.add_trace(go.Mesh3d(
                        x=x_sphere1, y=y_sphere1, z=z_sphere1,
                        i=i_faces, j=j_faces, k=k_faces,
                        color="black",  # Assign a random color for each sphere
                        opacity=1,
                        showlegend=False
                    ),
                        row=row, col=col)
                    # Update scene layout for each subplot
                    fig = plot_continent_outlines_on_sphere(fig, c=col, r=row)
                    col += 1
                    if col > 3:
                        col = 1
                        row += 1

                fig.update_layout(
                    title_text="Bagplot from different views",
                    height=800,  # Adjust height as needed
                    width=1000,  # Adjust width as needed
                    margin=dict(l=50, r=50, b=50, t=100),
                    showlegend=False,
                    plot_bgcolor='white',  # Background of the entire plot area
                    paper_bgcolor='white',  # Background outside the plot area
                    scene_camera={"eye": {'x': 1.25*contcenters_xyz[0][0], 'y': 1.25*contcenters_xyz[0][1], 'z': 1.25*contcenters_xyz[0][2]}},
                    scene_xaxis_visible=False,
                    scene_yaxis_visible=False,
                    scene_zaxis_visible=False,
                    scene_aspectmode="cube",
                    scene_bgcolor="rgba(0,0,0,0)",
                    scene2_camera={"eye": {'x': 1.25*contcenters_xyz[1][0], 'y': 1.25*contcenters_xyz[1][1], 'z': 1.25*contcenters_xyz[1][2]}},
                    scene2_xaxis_visible=False,
                    scene2_yaxis_visible=False,
                    scene2_zaxis_visible=False,
                    scene2_aspectmode="cube",
                    scene2_bgcolor="rgba(0,0,0,0)",
                    scene3_camera={"eye": {'x': 1.25*contcenters_xyz[2][0], 'y': 1.25*contcenters_xyz[2][1], 'z': 1.25*contcenters_xyz[2][2]}},
                    scene3_xaxis_visible=False,
                    scene3_yaxis_visible=False,
                    scene3_zaxis_visible=False,
                    scene3_aspectmode="cube",
                    scene3_bgcolor="rgba(0,0,0,0)",
                    scene4_camera={"eye": {'x': 1.25*contcenters_xyz[3][0], 'y': 1.25*contcenters_xyz[3][1], 'z': 1.25*contcenters_xyz[3][2]}},
                    scene4_xaxis_visible=False,
                    scene4_yaxis_visible=False,
                    scene4_zaxis_visible=False,
                    scene4_aspectmode="cube",
                    scene4_bgcolor="rgba(0,0,0,0)",
                    scene5_camera={"eye": {'x': 1.25*contcenters_xyz[4][0], 'y': 1.25*contcenters_xyz[4][1], 'z': 1.25*contcenters_xyz[4][2]}},
                    scene5_xaxis_visible=False,
                    scene5_yaxis_visible=False,
                    scene5_zaxis_visible=False,
                    scene5_aspectmode="cube",
                    scene5_bgcolor="rgba(0,0,0,0)",
                    scene6_camera={"eye": {'x': 1.25*contcenters_xyz[5][0], 'y': 1.25*contcenters_xyz[5][1], 'z': 1.25*contcenters_xyz[5][2]}},
                    scene6_xaxis_visible=False,
                    scene6_yaxis_visible=False,
                    scene6_zaxis_visible=False,
                    scene6_aspectmode="cube",
                    scene6_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family=font,
                        size=25)

                )

                fig.show()


            else:
                titles = ['Europe', 'Asia', "Africa", 'North America', 'South America', 'Australia'
                         ]
                image_names = ["bageurope.png", "bagasia.png", "bagafrica.png", "bagnamerica.png", "bagsamerica.png", "bagaustralia.png"]
                cameras = [{"eye": {'x': 1.25*contcenters_xyz[0][0], 'y': 1.25*contcenters_xyz[0][1], 'z': 1.25*contcenters_xyz[0][2]}},\
                           {"eye": {'x': 1.25*contcenters_xyz[1][0], 'y': 1.25*contcenters_xyz[1][1], 'z': 1.25*contcenters_xyz[1][2]}},\
                           {"eye": {'x': 1.25*contcenters_xyz[2][0], 'y': 1.25*contcenters_xyz[2][1], 'z': 1.25*contcenters_xyz[2][2]}},\
                           {"eye": {'x': 1.25*contcenters_xyz[3][0], 'y': 1.25*contcenters_xyz[3][1], 'z': 1.25*contcenters_xyz[3][2]}},\
                           {"eye": {'x': 1.25*contcenters_xyz[4][0], 'y': 1.25*contcenters_xyz[4][1], 'z': 1.25*contcenters_xyz[4][2]}},
                           {"eye": {'x': 1.25*contcenters_xyz[5][0], 'y': 1.25*contcenters_xyz[5][1], 'z': 1.25*contcenters_xyz[5][2]}}]

                for a in range(6):
                    fig = go.Figure(
                        data=[
                            go.Surface(x=x_sphere_rotated, y=y_sphere_rotated, z=z_sphere_rotated,
                                       surfacecolor=surface_colors_data,  # <--- Pass your conditional data here
                                       colorscale=custom_colorscale,  # <--- Apply your custom colorscale
                                       cmin=0, cmax=1,  # <--- Define the range of values in surface_colors_data
                                       opacity=1,
                                       showscale=False,  # Hide the color scale bar
                                       name='Sphere Regions')]
                    )


                    for i in range(len(data_x)):
                        center = [data_x[i], data_y[i], data_z[i]]
                        # Generate sphere mesh data for the current center
                        x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                            center, 0.005, 15, 15
                        )

                        # Add a Mesh3d trace for this sphere
                        fig.add_trace(go.Mesh3d(
                            x=x_sphere1, y=y_sphere1, z=z_sphere1,
                            i=i_faces, j=j_faces, k=k_faces,
                            color="black",  # Assign a random color for each sphere
                            opacity=1,
                            showlegend=False
                        ))


                    center = [mid[0], mid[1], mid[2]]
                    # Generate sphere mesh data for the current center
                    x_sphere1, y_sphere1, z_sphere1, i_faces, j_faces, k_faces = fun.generate_sphere_data(
                        center, 0.05, 15, 15
                    )

                    # Add a Mesh3d trace for this sphere
                    fig.add_trace(go.Mesh3d(
                        x=x_sphere1, y=y_sphere1, z=z_sphere1,
                        i=i_faces, j=j_faces, k=k_faces,
                        color="black",  # Assign a random color for each sphere
                        opacity=1,
                        showlegend=False
                    ))
                    # Update layout
                    fig.update_layout(
                        title=titles[a],
                        title_x=0.5,
                        scene=dict(
                            xaxis_visible=False,
                            yaxis_visible=False,
                            zaxis_visible=False,
                            bgcolor="rgba(0,0,0,0)",
                            aspectmode="cube",
                            camera=cameras[a]
                        ),
                        margin=dict(l=0, r=0, b=0, t=50),
                        font=dict(
                            family=font,
                            size=25)
                    )
                    fig=plot_continent_outlines_on_sphere(fig)
                    fig.write_image(image_names[a], engine="kaleido")

                fun.combine_images(image_names, figname, 2, 3)




