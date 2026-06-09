import math

from trabalho_pi.sa_heuristic import Solution, simulated_annealing


def create_test_solution() -> Solution:
    n = 4
    # Simple flow
    flow_matrix: list[list[float]] = [
        [0.0, 10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, 20.0, 0.0],
    ]
    # Simple dist
    dist_matrix: list[list[float]] = [
        [0.0, 2.0, 10.0, 12.0],
        [2.0, 0.0, 12.0, 14.0],
        [10.0, 12.0, 0.0, 3.0],
        [12.0, 14.0, 3.0, 0.0],
    ]
    chi = 1.0
    alpha = 0.5
    delta = 1.0

    # Assume hubs are 0 and 2.
    # 1 is allocated to 0, 3 is allocated to 2
    allocations = [0, 0, 2, 2]

    return Solution(n, flow_matrix, dist_matrix, chi, alpha, delta, allocations)


def test_solution_cost() -> None:
    sol = create_test_solution()
    # Cost calculation:
    # w_01=10 path: 0->0->0->1 cost=10*(1*0 + 0.5*0 + 1*2)=20
    # w_10=10 path: 1->0->0->0 cost=10*(1*2 + 0.5*0 + 1*0)=20
    # w_23=20 path: 2->2->2->3 cost=20*(1*0 + 0.5*0 + 1*3)=60
    # w_32=20 path: 3->2->2->2 cost=20*(1*3 + 0.5*0 + 1*0)=60
    # Total = 160
    assert math.isclose(sol.cost, 160.0)


def test_eval_swap() -> None:
    sol = create_test_solution()

    # Let's swap node 1 to hub 2
    _ = sol.eval_swap(1, 2)
    sol.apply_swap(1, 2)

    # Verify delta matches recalculation
    recalculated_cost = sol.calculate_total_cost()
    assert math.isclose(sol.cost, recalculated_cost)


def test_eval_move() -> None:
    sol = create_test_solution()

    # Hub 0 cluster is {0, 1}. Move hub to 1.
    _ = sol.eval_move(0, 1)
    sol.apply_move(0, 1)

    recalculated_cost = sol.calculate_total_cost()
    assert math.isclose(sol.cost, recalculated_cost)
    assert 1 in sol.get_hubs()
    assert 0 not in sol.get_hubs()


def test_eval_singleton_move() -> None:
    sol = create_test_solution()

    # First, let's create a singleton.
    # Make node 1 its own hub. So hubs are 0, 1, 2. (This is p=3 temporarily)
    sol.allocations = [0, 1, 2, 2]
    sol.cost = sol.calculate_total_cost()

    # Singleton move on hub 1: it was a singleton.
    # Let's make node 3 the new hub, and allocate node 1 to hub 2.
    _ = sol.eval_singleton_move(1, 3, 2)
    sol.apply_singleton_move(1, 3, 2)

    recalculated_cost = sol.calculate_total_cost()
    assert math.isclose(sol.cost, recalculated_cost)
    assert sol.allocations[1] == 2
    assert sol.allocations[3] == 3
    assert 3 in sol.get_hubs()


def test_simulated_annealing() -> None:
    n = 4
    p = 2
    flow_matrix: list[list[float]] = [
        [0.0, 10.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, 20.0, 0.0],
    ]
    dist_matrix: list[list[float]] = [
        [0.0, 2.0, 10.0, 12.0],
        [2.0, 0.0, 12.0, 14.0],
        [10.0, 12.0, 0.0, 3.0],
        [12.0, 14.0, 3.0, 0.0],
    ]
    chi = 1.0
    alpha = 0.5
    delta = 1.0

    # The SA should find the optimal clustering easily
    best_sol = simulated_annealing(
        n, p, flow_matrix, dist_matrix, chi, alpha, delta, seed=42
    )

    # Cost of best solution should be 160.0
    assert best_sol.cost <= 160.0 + 1e-5

    # Check that it has exactly 2 hubs
    assert len(best_sol.get_hubs()) == 2
