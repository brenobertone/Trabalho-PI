import math
import random
import statistics
from typing import Dict, List, Optional


class Solution:
    def __init__(
        self,
        n: int,
        flow_matrix: List[List[float]],
        dist_matrix: List[List[float]],
        chi: float,
        alpha: float,
        delta: float,
        allocations: Optional[List[int]] = None,
    ):
        self.n = n
        self.flow_matrix = flow_matrix
        self.dist_matrix = dist_matrix
        self.chi = chi
        self.alpha = alpha
        self.delta = delta

        if allocations is None:
            self.allocations = list(range(n))  # Default, invalid for p < n
        else:
            self.allocations = allocations[:]

        self.cost = self.calculate_total_cost()

    def calculate_total_cost(self) -> float:
        cost = 0.0
        for i in range(self.n):
            for j in range(self.n):
                hub_i = self.allocations[i]
                hub_j = self.allocations[j]
                w = self.flow_matrix[i][j]
                if w > 0:
                    d_i_hub = self.dist_matrix[i][hub_i]
                    d_hub_hub = self.dist_matrix[hub_i][hub_j]
                    d_hub_j = self.dist_matrix[hub_j][j]
                    cost += w * (
                        self.chi * d_i_hub
                        + self.alpha * d_hub_hub
                        + self.delta * d_hub_j
                    )
        return cost

    def copy(self) -> "Solution":
        return Solution(
            self.n,
            self.flow_matrix,
            self.dist_matrix,
            self.chi,
            self.alpha,
            self.delta,
            self.allocations,
        )

    def get_hubs(self) -> List[int]:
        return list(set(self.allocations))

    def get_clusters(self) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {}
        for i, hub in enumerate(self.allocations):
            if hub not in clusters:
                clusters[hub] = []
            clusters[hub].append(i)
        return clusters

    def eval_swap(self, node: int, new_hub: int) -> float:
        old_hub = self.allocations[node]
        if old_hub == new_hub:
            return 0.0

        delta_val = 0.0
        # Flow from node to others
        for j in range(self.n):
            w = self.flow_matrix[node][j]
            if w > 0:
                hub_j = self.allocations[j] if j != node else new_hub
                old_cost = w * (
                    self.chi * self.dist_matrix[node][old_hub]
                    + self.alpha * self.dist_matrix[old_hub][self.allocations[j]]
                    + self.delta * self.dist_matrix[self.allocations[j]][j]
                )
                new_cost = w * (
                    self.chi * self.dist_matrix[node][new_hub]
                    + self.alpha * self.dist_matrix[new_hub][hub_j]
                    + self.delta * self.dist_matrix[hub_j][j]
                )
                delta_val += new_cost - old_cost

        # Flow from others to node
        for i in range(self.n):
            if i == node:
                continue
            w = self.flow_matrix[i][node]
            if w > 0:
                hub_i = self.allocations[i]
                old_cost = w * (
                    self.chi * self.dist_matrix[i][hub_i]
                    + self.alpha * self.dist_matrix[hub_i][old_hub]
                    + self.delta * self.dist_matrix[old_hub][node]
                )
                new_cost = w * (
                    self.chi * self.dist_matrix[i][hub_i]
                    + self.alpha * self.dist_matrix[hub_i][new_hub]
                    + self.delta * self.dist_matrix[new_hub][node]
                )
                delta_val += new_cost - old_cost

        return delta_val

    def apply_swap(self, node: int, new_hub: int) -> float:
        delta_val = self.eval_swap(node, new_hub)
        self.allocations[node] = new_hub
        self.cost += delta_val
        return delta_val

    def eval_move(self, old_hub: int, new_hub: int) -> float:
        delta_val = 0.0
        cluster = [i for i in range(self.n) if self.allocations[i] == old_hub]

        for c in cluster:
            # flow from c to anywhere
            for j in range(self.n):
                w = self.flow_matrix[c][j]
                if w > 0:
                    hub_j = self.allocations[j]
                    new_hub_j = new_hub if hub_j == old_hub else hub_j
                    old_cost = w * (
                        self.chi * self.dist_matrix[c][old_hub]
                        + self.alpha * self.dist_matrix[old_hub][hub_j]
                        + self.delta * self.dist_matrix[hub_j][j]
                    )
                    new_cost = w * (
                        self.chi * self.dist_matrix[c][new_hub]
                        + self.alpha * self.dist_matrix[new_hub][new_hub_j]
                        + self.delta * self.dist_matrix[new_hub_j][j]
                    )
                    delta_val += new_cost - old_cost

            # flow from anywhere to c
            for i in range(self.n):
                if self.allocations[i] == old_hub:
                    continue  # already processed above if i is in cluster
                w = self.flow_matrix[i][c]
                if w > 0:
                    hub_i = self.allocations[i]
                    old_cost = w * (
                        self.chi * self.dist_matrix[i][hub_i]
                        + self.alpha * self.dist_matrix[hub_i][old_hub]
                        + self.delta * self.dist_matrix[old_hub][c]
                    )
                    new_cost = w * (
                        self.chi * self.dist_matrix[i][hub_i]
                        + self.alpha * self.dist_matrix[hub_i][new_hub]
                        + self.delta * self.dist_matrix[new_hub][c]
                    )
                    delta_val += new_cost - old_cost
        return delta_val

    def apply_move(self, old_hub: int, new_hub: int) -> float:
        delta_val = self.eval_move(old_hub, new_hub)
        for i in range(self.n):
            if self.allocations[i] == old_hub:
                self.allocations[i] = new_hub
        self.cost += delta_val
        return delta_val

    def eval_singleton_move(
        self, old_hub: int, new_hub: int, new_hub_allocation: int
    ) -> float:
        delta_val = 0.0
        affected_nodes = [old_hub, new_hub]

        for c in affected_nodes:
            for j in range(self.n):
                w = self.flow_matrix[c][j]
                if w > 0:
                    hub_j = self.allocations[j]
                    old_cost = w * (
                        self.chi * self.dist_matrix[c][self.allocations[c]]
                        + self.alpha * self.dist_matrix[self.allocations[c]][hub_j]
                        + self.delta * self.dist_matrix[hub_j][j]
                    )

                    new_hub_c = new_hub_allocation if c == old_hub else new_hub
                    new_hub_j = hub_j
                    if j == old_hub:
                        new_hub_j = new_hub_allocation
                    if j == new_hub:
                        new_hub_j = new_hub

                    new_cost = w * (
                        self.chi * self.dist_matrix[c][new_hub_c]
                        + self.alpha * self.dist_matrix[new_hub_c][new_hub_j]
                        + self.delta * self.dist_matrix[new_hub_j][j]
                    )
                    delta_val += new_cost - old_cost

            for i in range(self.n):
                if i in affected_nodes:
                    continue
                w = self.flow_matrix[i][c]
                if w > 0:
                    hub_i = self.allocations[i]
                    old_cost = w * (
                        self.chi * self.dist_matrix[i][hub_i]
                        + self.alpha * self.dist_matrix[hub_i][self.allocations[c]]
                        + self.delta * self.dist_matrix[self.allocations[c]][c]
                    )

                    new_hub_c = new_hub_allocation if c == old_hub else new_hub
                    new_cost = w * (
                        self.chi * self.dist_matrix[i][hub_i]
                        + self.alpha * self.dist_matrix[hub_i][new_hub_c]
                        + self.delta * self.dist_matrix[new_hub_c][c]
                    )
                    delta_val += new_cost - old_cost

        return delta_val

    def apply_singleton_move(
        self, old_hub: int, new_hub: int, new_hub_allocation: int
    ) -> float:
        delta_val = self.eval_singleton_move(old_hub, new_hub, new_hub_allocation)
        self.allocations[old_hub] = new_hub_allocation
        self.allocations[new_hub] = new_hub
        self.cost += delta_val
        return delta_val


def simulated_annealing(
    n: int,
    p: int,
    flow_matrix: List[List[float]],
    dist_matrix: List[List[float]],
    chi: float,
    alpha: float,
    delta: float,
    seed: int = 42,
) -> Solution:
    random.seed(seed)

    # 1. Initial solution
    hubs = random.sample(range(n), p)
    allocations = [0] * n
    for i in range(n):
        if i in hubs:
            allocations[i] = i
        else:
            best_hub = min(hubs, key=lambda h: dist_matrix[i][h])
            allocations[i] = best_hub

    current = Solution(n, flow_matrix, dist_matrix, chi, alpha, delta, allocations)
    best = current.copy()

    # 2. T0 estimation
    deltas = []
    for _ in range(100):
        clusters = current.get_clusters()
        hubs_list = list(clusters.keys())
        node = random.randint(0, n - 1)
        if current.allocations[node] != node:
            possible_hubs = [h for h in hubs_list if h != current.allocations[node]]
            if possible_hubs:
                new_hub = random.choice(possible_hubs)
                d = current.eval_swap(node, new_hub)
                if d > 0:
                    deltas.append(d)

    if len(deltas) > 1:
        t0 = statistics.stdev(deltas)
    elif len(deltas) == 1:
        t0 = deltas[0]
    else:
        t0 = 100.0

    if t0 < 1e-5:
        t0 = 100.0

    t_current = t0
    t_reheat = t0

    p_swap = 0.40
    p_move = 0.60

    chain_length = 2 * n * p
    cooling_factor = 0.97

    r = 0
    c = 0

    while r <= 4:
        while c <= 5:
            for _ in range(chain_length):
                if t_current < 1e-10:
                    t_current = 1e-10

                clusters = current.get_clusters()
                hubs_list = list(clusters.keys())

                op = random.choices(["swap", "move"], weights=[p_swap, p_move])[0]

                if op == "swap":
                    non_hubs = [i for i in range(n) if current.allocations[i] != i]
                    if not non_hubs:
                        continue
                    node = random.choice(non_hubs)
                    old_hub = current.allocations[node]
                    possible_hubs = [h for h in hubs_list if h != old_hub]
                    if not possible_hubs:
                        continue
                    new_hub = random.choice(possible_hubs)

                    delta_cost = current.eval_swap(node, new_hub)

                    if delta_cost < 0 or random.random() < math.exp(
                        -delta_cost / t_current
                    ):
                        current.apply_swap(node, new_hub)
                        if current.cost < best.cost:
                            best = current.copy()
                            c = 0

                elif op == "move":
                    old_hub = random.choice(hubs_list)
                    cluster_nodes = clusters[old_hub]
                    non_hubs_in_cluster = [x for x in cluster_nodes if x != old_hub]

                    if non_hubs_in_cluster:
                        new_hub = random.choice(non_hubs_in_cluster)
                        delta_cost = current.eval_move(old_hub, new_hub)
                        if delta_cost < 0 or random.random() < math.exp(
                            -delta_cost / t_current
                        ):
                            current.apply_move(old_hub, new_hub)
                            if current.cost < best.cost:
                                best = current.copy()
                                c = 0
                    else:
                        other_clusters = [h for h in hubs_list if h != old_hub]
                        if not other_clusters:
                            continue

                        candidates = [
                            x for x in range(n) if current.allocations[x] != x
                        ]
                        if not candidates:
                            continue
                        new_hub = random.choice(candidates)

                        new_hub_allocation = random.choice(other_clusters)

                        delta_cost = current.eval_singleton_move(
                            old_hub, new_hub, new_hub_allocation
                        )
                        if delta_cost < 0 or random.random() < math.exp(
                            -delta_cost / t_current
                        ):
                            current.apply_singleton_move(
                                old_hub, new_hub, new_hub_allocation
                            )
                            if current.cost < best.cost:
                                best = current.copy()
                                c = 0

            t_current *= cooling_factor
            c += 1

        t_current = t_reheat
        p_swap = min(1.0, p_swap + 0.05)
        p_move = 1.0 - p_swap
        r += 1
        c = 0

    return best
