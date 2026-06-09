import math
from typing import List, Tuple


def compute_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Computes the Euclidean distance matrix for a list of 2D coordinates.

    Args:
        coords: List of (x, y) coordinates for each node.

    Returns:
        A 2D list (matrix) where matrix[i][j] is the Euclidean
        distance between node i and node j.
    """
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                matrix[i][j] = math.sqrt(dx * dx + dy * dy)
    return matrix


def compute_node_flows(
    flow_matrix: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """
    Computes total supply (O_i) and demand (D_i) for each node based on flow matrix.

    Args:
        flow_matrix: A 2D list where flow_matrix[i][j] is the flow
        from node i to node j (W_ij).

    Returns:
        A tuple (O_flows, D_flows) where:
            O_flows is a list of total supply for each node.
            D_flows is a list of total demand for each node.
    """
    n = len(flow_matrix)
    if n == 0:
        return [], []

    o_flows = [0.0] * n
    d_flows = [0.0] * n

    for i in range(n):
        for j in range(n):
            o_flows[i] += flow_matrix[i][j]
            d_flows[j] += flow_matrix[i][j]

    return o_flows, d_flows
