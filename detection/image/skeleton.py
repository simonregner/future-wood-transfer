import cv2
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize, thin, medial_axis
from skimage.filters import frangi

from numba import njit


def contour_to_mask(contour, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def get_skeleton(mask):
    skeleton = thin(mask > 0)
    #skeleton = medial_axis(mask > 0, ).astype(np.uint16)
    return skeleton.astype(np.uint8)

def skeleton_to_graph(skeleton):
    graph = sknw.build_sknw(skeleton)
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

def find_longest_path_from_start(graph, start_node):
    """
    Finds the longest shortest path starting from the given start_node.
    """
    longest_path = []
    max_length = 0

    # Use Dijkstra's algorithm to get all shortest paths from start_node
    lengths, paths = nx.single_source_dijkstra(graph, source=start_node, weight='weight')

    # Find the node with the longest shortest path from start_node
    farthest_node = max(lengths, key=lengths.get)
    longest_path = paths[farthest_node]

    return longest_path

def get_normals(centerline):
    tangents = np.gradient(centerline.astype(np.float32), axis=0)
    normals = np.zeros_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals


def remove_bottom_corner_nodes(graph, mask, margin_y=50):
    height, width = mask.shape
    nodes_to_remove = []
    for node, data in graph.nodes(data=True):
        y, x = map(int, data['o'])
        # Only remove nodes that are in the mask (255 region) at bottom-left or bottom-right
        if y > height - margin_y and mask[y, x] == 255:
            nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)

    return graph

def get_nearest_node(graph, car_point):
    existing_nodes = np.array([graph.nodes[n]['o'] for n in graph.nodes])
    distances = np.linalg.norm(existing_nodes - car_point, axis=1)
    nearest_node_idx = np.argmin(distances)
    nearest_node = list(graph.nodes())[nearest_node_idx]

    return nearest_node_idx, nearest_node

def add_start_node(graph, car_point, nearest_node_idx):
    bottom_y, bottom_center_x = car_point

    # Add bottom-center node explicitly to graph
    new_node_idx = max(graph.nodes()) + 1
    graph.add_node(new_node_idx, o=car_point)

    #nearest_node_idx, nearest_node = get_nearest_node(graph, car_point)

    # Connect explicitly to nearest node
    graph.add_edge(new_node_idx, nearest_node_idx, weight=1.0, pts=np.array([
        car_point,
        graph.nodes[nearest_node_idx]['o']
    ]), dtype=int)

    return graph, new_node_idx

def remove_nearest_nodes_ends(graph, nearest_existing_node):
    print("Nearst Nodes: ", len(graph.edges(nearest_existing_node)))
    if len(graph.edges(nearest_existing_node)) >= 3:
        # Get all edges connected to the node
        edges_to_check = list(graph.edges(nearest_existing_node))

        # Identify edges that go downward to left or right
        edges_to_remove = []
        nodes_to_remove = []
        node_y, node_x = graph.nodes[nearest_existing_node]['o']

        for s, e in edges_to_check:
            neighbor = e if s == nearest_existing_node else s
            neighbor_y, neighbor_x = graph.nodes[neighbor]['o']

            # Check if the edge goes downward (y increasing) and sideways (x significantly changing)
            if neighbor_y > node_y and (neighbor_x < node_x or neighbor_x > node_x):
                edges_to_remove.append((s, e))
                nodes_to_remove.append(neighbor)

        # Keep only 1 connection (the one closest to the car POV node)
        if len(edges_to_remove) > 1:
            print("Removing bottom-left & bottom-right edges: ", len(edges_to_remove))
            edges_to_remove = edges_to_remove[:2]  # Remove two edges
            nodes_to_remove = nodes_to_remove[:2]  # Remove corresponding nodes

        # Remove these unnecessary edges and nodes
        graph.remove_edges_from(edges_to_remove)
        graph.remove_nodes_from(nodes_to_remove)

    return graph


def extend_path(graph, mask, longest_path_nodes):
    """
    Extends the graph in the last edge direction to the last 255-pixel in the mask.
    """
    start_node = longest_path_nodes[-2]  # Second last node
    end_node = longest_path_nodes[-1]  # Last node (current end point)

    # Get last edge direction
    start_y, start_x = graph.nodes[start_node]['o']
    end_y, end_x = graph.nodes[end_node]['o']

    direction = np.array([end_y - start_y, end_x - start_x])  # Vector direction
    direction = direction / np.linalg.norm(direction)  # Normalize

    # Extend until we find the last 255 pixel
    max_distance = 100  # Limit the extension range
    new_end_y, new_end_x = end_y, end_x

    for _ in range(max_distance):
        new_end_y += direction[0]
        new_end_x += direction[1]

        # Ensure valid image coordinates
        new_end_y = int(min(max(new_end_y, 0), mask.shape[0] - 1))
        new_end_x = int(min(max(new_end_x, 0), mask.shape[1] - 1))

        if mask[new_end_y, new_end_x] != 255:
            break  # Stop if outside road mask

    # Add new extended node
    new_end_node_idx = max(graph.nodes()) + 1
    graph.add_node(new_end_node_idx, o=np.array([new_end_y, new_end_x]))

    # Connect to previous last node
    graph.add_edge(end_node, new_end_node_idx, weight=1.0, pts=np.array([
        np.array(graph.nodes[end_node]['o']),
        np.array(graph.nodes[new_end_node_idx]['o'])
    ]))

    return graph  # Return new endpoint