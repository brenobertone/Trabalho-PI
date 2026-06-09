import math

from trabalho_pi.utils import compute_distance_matrix, compute_node_flows


def test_compute_distance_matrix() -> None:
    coords = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

    dist_matrix = compute_distance_matrix(coords)

    assert len(dist_matrix) == 3
    assert len(dist_matrix[0]) == 3

    # Distance to self is 0
    assert dist_matrix[0][0] == 0.0
    assert dist_matrix[1][1] == 0.0

    # Check known distances (3-4-5 triangle)
    assert math.isclose(dist_matrix[0][1], 5.0)
    assert math.isclose(dist_matrix[1][0], 5.0)

    # Distance between (0,0) and (6,8) is 10
    assert math.isclose(dist_matrix[0][2], 10.0)
    assert math.isclose(dist_matrix[2][0], 10.0)

    # Distance between (3,4) and (6,8) is 5
    assert math.isclose(dist_matrix[1][2], 5.0)
    assert math.isclose(dist_matrix[2][1], 5.0)


def test_compute_node_flows() -> None:
    # flow_matrix[i][j] = flow from i to j
    flow_matrix = [
        [0.0, 10.0, 20.0],  # Node 0 sends 10 to 1, 20 to 2. Total O_0 = 30
        [5.0, 0.0, 15.0],  # Node 1 sends 5 to 0, 15 to 2. Total O_1 = 20
        [2.0, 3.0, 0.0],  # Node 2 sends 2 to 0, 3 to 1. Total O_2 = 5
    ]

    o_flows, d_flows = compute_node_flows(flow_matrix)

    # Check Supply (O_i = sum over j of W_ij)
    assert len(o_flows) == 3
    assert o_flows[0] == 30.0
    assert o_flows[1] == 20.0
    assert o_flows[2] == 5.0

    # Check Demand (D_i = sum over j of W_ji)
    assert len(d_flows) == 3
    assert d_flows[0] == 5.0 + 2.0  # 7.0
    assert d_flows[1] == 10.0 + 3.0  # 13.0
    assert d_flows[2] == 20.0 + 15.0  # 35.0


def test_compute_node_flows_empty() -> None:
    o_flows, d_flows = compute_node_flows([])
    assert o_flows == []
    assert d_flows == []
