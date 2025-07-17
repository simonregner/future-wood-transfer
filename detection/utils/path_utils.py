import numpy as np
from sklearn.decomposition import PCA

def calculate_fitted_line(points, degree, t_fit):
    # PCA to sort points
    t = PCA(n_components=1).fit_transform(points).flatten()
    sorted_idx = np.argsort(t)
    points_sorted = points[sorted_idx]
    t_sorted = t[sorted_idx]

    # Fit polynomial for each coordinate using t_sorted as the parameter:
    # (you might want to adjust degrees as needed)
    poly_x = np.polyfit(t_sorted, points_sorted[:, 0], degree)
    poly_y = np.polyfit(t_sorted, points_sorted[:, 1], 2)
    poly_z = np.polyfit(t_sorted, points_sorted[:, 2], degree)

    # Generate the main fitted t-values for your current range:
    t_fit_main = np.linspace(t_sorted[0], t_sorted[-1], t_fit)
    x_fit_main = np.polyval(poly_x, t_fit_main)
    y_fit_main = np.polyval(poly_y, t_fit_main)
    # y_fit_main = np.zeros(t_fit_len)
    z_fit_main = np.polyval(poly_z, t_fit_main)

    # Estimate a spacing for the t-values.
    # One simple approach is to use the difference between the first two sorted t-values.
    # (You may also compute the median of t differences if that's more robust.)
    dt = t_sorted[1] - t_sorted[0]

    n_extension = 0  # Number of extra points to extend before the start of the data

    # Generate n_extension extra t-values that extend *before* the beginning of your data.
    # For example, if you have 5 extra points, you can create them from t_sorted[0] - 5*dt up to t_sorted[0]
    t_fit_ext = np.linspace(t_sorted[0] - n_extension * dt, t_sorted[0], n_extension, endpoint=False)

    # Evaluate the fitted polynomial on the extension t-values:
    x_fit_ext = np.polyval(poly_x, t_fit_ext)
    y_fit_ext = np.polyval(poly_y, t_fit_ext)
    z_fit_ext = np.polyval(poly_z, t_fit_ext)

    # Option 2: If you want to combine the extension and the fitted points:
    x_fit = np.concatenate([x_fit_ext, x_fit_main])
    y_fit = np.concatenate([y_fit_ext, y_fit_main])
    z_fit = np.concatenate([z_fit_ext, z_fit_main])

    return x_fit, y_fit, z_fit

def ensure_first_point_closest_to_origin(points) -> np.array:
    points = list(points)  # in case it's a numpy array or tuple
    first = np.array(points[0])
    last = np.array(points[-1])
    origin = np.array([0, 0, 0])

    if np.linalg.norm(last - origin) < np.linalg.norm(first - origin):
        return points[::-1]  # reversed

    return points