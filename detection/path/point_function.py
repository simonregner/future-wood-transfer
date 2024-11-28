import numpy as np
from scipy.optimize import curve_fit

def fit_line_3d(points, degree=5):
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

