import math

def get_road_width(boundary_1, boundary_2):
    x1, y1, z1 = boundary_1[0]
    x2, y2, z2 = boundary_2[0]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)