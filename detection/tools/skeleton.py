import cv2
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize


def contour_to_mask(contour, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def get_skeleton(mask):
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton):
    graph = sknw.build_sknw(skeleton, multi=False)
    return graph


def longest_path_in_graph(graph):
    longest_path = []
    max_length = 0

    for n1 in graph.nodes():
        for n2 in graph.nodes():
            if n1 != n2:
                try:
                    path = nx.shortest_path(graph, n1, n2, weight='weight')
                    length = nx.path_weight(graph, path, weight='weight')
                    if length > max_length:
                        longest_path = path
                        max_length = length
                except nx.NetworkXNoPath:
                    continue

    coords = [graph.nodes[n]['o'] for n in longest_path]
    return np.array(coords)