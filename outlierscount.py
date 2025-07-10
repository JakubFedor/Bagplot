import fun
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import numpy as np
base = importr('base')



def points_on_semicircle(point1, point2, num_points):
        """
        Generates a grid of points for a semicircle on a unit sphere.
        The semicircle lies on the plane defined by point1, point2, and the origin.
        The first point of the grid will be point2, and the last will be its antipodal point.

        Args:
            point1 (list or np.array): A point on the unit sphere, helping to define the plane.
            point2 (list or np.array): The starting point of the semicircle on the unit sphere.
                                        The semicircle will end at its antipodal point (-point2).
            num_points (int): The number of points to generate along the semicircle.

        Returns:
            np.array: A 2D array where each row is a point [x, y, z] on the semicircle.
                      Returns an empty array if the plane cannot be uniquely defined
                      (i.e., point1, point2, and the origin are collinear).
        """
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)

        # 1. Normalize the input points (ensure they are on the unit sphere)
        p1 = p1 / np.linalg.norm(p1)
        p2 = p2 / np.linalg.norm(p2)

        # 2. Calculate the normal vector of the plane containing the semicircle
        # This plane is defined by point1, point2, and the origin.
        normal_vector = np.cross(p1, p2)

        # Check if points defining the plane are collinear with the origin
        if np.linalg.norm(normal_vector) < 1e-9:
            print("Warning: Input points (point1, point2) are collinear with the origin. Cannot form a unique plane.")
            return np.array([])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # 3. The starting point for rotation is p2.
        # We will rotate p2 around the normal_vector from 0 to pi radians.
        angles = np.linspace(0, np.pi, num_points)
        test1= p2 * np.cos(angles[1]) + \
                            np.cross(normal_vector, p2) * np.sin(angles[1]) + \
                            normal_vector * np.dot(normal_vector, p2) * (1 - np.cos(angles[1]))
        test2= p2 * np.cos(angles[-2]) + \
                            np.cross(normal_vector, p2) * np.sin(angles[-2]) + \
                            normal_vector * np.dot(normal_vector, p2) * (1 - np.cos(angles[-2]))

        semicircle_points = []
        if np.sign(np.dot(test2,point1))==np.sign(np.dot(test2,point2)) and np.sign(np.dot(test1,point1))==np.sign(np.dot(test1,point2)) and np.sign(np.dot(test1,point1))==np.sign(np.dot(test2,point1)) and np.sign(np.dot(test1,point2))==np.sign(np.dot(test2,point2)):
            for angle in angles:
                # Rodrigues' rotation formula: v_rot = v * cos(theta) + (k x v) * sin(theta) + k * (k . v) * (1 - cos(theta))
                # Here, v is p2, k is normal_vector, and theta is angle
                rotated_point = p2 * np.cos(angle) + \
                                np.cross(normal_vector, p2) * np.sin(angle) + \
                                normal_vector * np.dot(normal_vector, p2) * (1 - np.cos(angle))
                semicircle_points.append(rotated_point)
        else:
            for angle in angles:
                rotated_point = p2 * np.cos(-angle) + \
                                np.cross(normal_vector, p2) * np.sin(-angle) + \
                                normal_vector * np.dot(normal_vector, p2) * (1 - np.cos(-angle))
                semicircle_points.append(rotated_point)

        return np.array(semicircle_points)




def outliers_count(data, weights=None, dist="arc", a=0.99, borderdist="mean", res=500):
    data = np.asarray(data)
    # x, y and z values from the data
    data_x = data[:, 0]
    data_y = data[:, 1]
    data_z = data[:, 2]
    # loading the data into R instance
    fv_x = FloatVector(data_x)
    fv_y = FloatVector(data_y)
    fv_z = FloatVector(data_z)
    robjects.globalenv['x'] = fv_x
    robjects.globalenv['y'] = fv_y
    robjects.globalenv['z'] = fv_z

    if weights=None:
        weights = np.ones(len(data))
    weights = np.asarray(weights)
    fv_weights = FloatVector(weights)
    robjects.r("data=cbind(x,y,z)")
    robjects.globalenv['weights'] = fv_weights
    # calculate the depth of the datapoints
    datadepth = robjects.r("ahD(data,weights)")
    # finding the depth of the border of the bag
    borderdepth = np.median(datadepth)
    nonbagpoints= [data[i] for i in range(len(data_x)) if datadepth[i] < borderdepth]
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

    phi, theta = np.mgrid[0:np.pi:res * 1j, 0:2 * np.pi:res * 1j]
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

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

    surface_colors_data = x_sphere_rotated.copy()
    for i in range(len(x_sphere)):
        x_spherefv = FloatVector(x_sphere_rotated[i])
        y_spherefv = FloatVector(y_sphere_rotated[i])
        z_spherefv = FloatVector(z_sphere_rotated[i])
        robjects.globalenv['x'] = x_spherefv
        robjects.globalenv['y'] = y_spherefv
        robjects.globalenv['z'] = z_spherefv
        robjects.r("coords=cbind(x,y,z)")
        surface_colors_data[i] = np.where(np.array(robjects.r("ahD(data,weights,coords)")) >= borderdepth,
                                          1, 0)
    # finding the borders of the bag
    borders = []
    for i in range(len(x_sphere_rotated)):
        borders.append([np.where(surface_colors_data[:, i] == 0)[0][0] - 1, i])
    middist = x_sphere_rotated.copy()
    for i in range(len(x_sphere_rotated)):
        for j in range(len(x_sphere_rotated[i])):
            if dist == "arc":
                middist[i, j] = fun.arcdist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]],
                                            mid)
            elif dist == "cos":
                middist[i, j] = fun.cosdist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]],
                                            mid)
            else:
                middist[i, j] = fun.chorddist([x_sphere_rotated[i, j], y_sphere_rotated[i, j], z_sphere_rotated[i, j]],
                                              mid)
    # finding the maximal distance from the center of the bagplot to the border of the bag
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
    num_outliers=0
    for point in nonbagpoints:
        grid = points_on_semicircle(point, mid, res)
        grid_xfv = FloatVector(grid[:,0])
        grid_yfv = FloatVector(grid[:,1])
        grid_zfv = FloatVector(grid[:,2])
        robjects.globalenv['x'] = grid_xfv
        robjects.globalenv['y'] = grid_yfv
        robjects.globalenv['z'] = grid_zfv
        robjects.r("coords=cbind(x,y,z)")
        bagornot =np.where(np.array(robjects.r("ahD(data,weights,coords)")) >= borderdepth,1, 0)
        borderindex = np.where(bagornot == 0)[0][0] - 1

        if dist == "arc":
            borderd = fun.arcdist(grid[borderindex], mid)
            loopd = factor[0] * borderd
            pointdist = fun.arcdist(point, mid)
        elif dist == "cos":
            borderd = fun.cosdist(grid[borderindex], mid)[0]
            loopd = factor[0] * borderd
            pointdist = fun.cosdist(point, mid)
            borderd = fun.chorddist(grid[borderindex], mid)
        elif dist == "chord":
            loopd = factor[0] * borderd
            pointdist = fun.chorddist(point, mid)
        else:
            print("Wrong format of dist")
            return

        if pointdist>loopd:
            num_outliers+=1



    return num_outliers


