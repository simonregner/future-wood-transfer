import numpy as np
from scipy.optimize import curve_fit

from scipy.interpolate import splprep, splev

from scipy.interpolate import LSQUnivariateSpline

from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline

def fit_line_3d(points, degree=4):
    # Extract x, y, and z coordinates from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Use the parameter t as the distance along the points
    t = np.linspace(0, 1, len(points))


    # Define a function to fit polynomial curves of a given degree
    def poly_fit(t, *coeffs):
        degree = (len(coeffs) - 1) // 3
        x_coeffs = coeffs[:degree + 1]
        y_coeffs = coeffs[degree + 1: 2 * (degree + 1)]
        z_coeffs = coeffs[2 * (degree + 1):]
        x = sum(c * t ** i for i, c in enumerate(reversed(x_coeffs)))
        y = sum(c * t ** i for i, c in enumerate(reversed(y_coeffs)))
        z = sum(c * t ** i for i, c in enumerate(reversed(z_coeffs)))
        return x, y, z


    # Fit polynomial curves for x, y, and z with respect to t
    params_xyz, _ = curve_fit(lambda t, *coeffs: np.hstack(poly_fit(t, *coeffs)), t, np.hstack((x, y, z)),
                              p0=np.ones(3 * (degree + 1)))

    # Generate points on the fitted curve
    t_fine = np.linspace(0, 1, 100)
    x_fine, y_fine, z_fine = poly_fit(t_fine, *params_xyz)

    return x_fine, y_fine, z_fine

def fit_line_3d_smooth(points, smoothing_factor=0):
    """
    Fit a smooth 3D line to the given points using a cubic B-spline.

    Parameters:
        points (numpy.ndarray): Array of shape (N, 3) representing the 3D points.
        smoothing_factor (float): Parameter controlling the smoothness of the spline.
                                   Higher values result in smoother curves.

    Returns:
        numpy.ndarray: Fitted x, y, and z coordinates.
    """
    # Extract x, y, and z coordinates from points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create a B-spline representation of the curve
    tck, u = splprep([x, y, z], s=smoothing_factor)

    # Generate smoothed points along the fitted spline
    u_fine = np.linspace(0, 1, 100)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    return x_fine, y_fine, z_fine


def func(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y

def fit_line_3d_smooth_new(points):
    # Extract X, Y, Z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Fit a spline through the points
    tck, u = splprep([x, y, z], s=20)  # s is a smoothing factor, adjust as needed
    u_fine = np.linspace(0, 1, 1000)  # Parameter values for interpolation
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    # Flatten the grid to create 3D points
    points_smooth = np.column_stack([x_fine.ravel(), y_fine.ravel(), z_fine.ravel()])

    return points_smooth