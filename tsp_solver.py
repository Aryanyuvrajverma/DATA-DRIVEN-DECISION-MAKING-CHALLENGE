"""
High-Performance Traveling Salesman Problem Solver.
Pure Python + NumPy implementation.
NO third-party optimization libraries used.
Includes stepwise generators for live animation in UI.
"""

import time
import math
import random
import numpy as np


def read_tsp(file_path):
    """Parses a standard TSPLIB file (.tsp) and extracts node coordinates."""
    coords = []
    node_ids = []
    name = "TSP"
    in_coord_section = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("NAME"):
                parts = line.split(":")
                if len(parts) > 1:
                    name = parts[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                in_coord_section = True
                continue
            elif line.startswith("EOF"):
                break
            elif in_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        node_ids.append(nid)
                        coords.append([x, y])
                    except ValueError:
                        pass
    return name, np.array(coords, dtype=np.float64), node_ids


def compute_distance_matrix(coords, round_integers=True):
    """
    Computes pairwise Euclidean distance matrix.
    TSPLIB EUC_2D uses rounded integers: nint(sqrt(dx^2 + dy^2)).
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    if round_integers:
        return np.round(dist).astype(np.float64)
    return dist


def calculate_tour_length(tour, dist_matrix):
    """Calculates the total round-trip distance for a tour."""
    return float(np.sum(dist_matrix[tour, np.roll(tour, -1)]))


# -------------------------------------------------------------
# 1. Construction Heuristics
# -------------------------------------------------------------

def nearest_neighbor(dist_matrix, start_node=0):
    """Classic Nearest Neighbor heuristic starting from start_node."""
    n = dist_matrix.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start_node] = False
    tour = np.zeros(n, dtype=np.int32)
    tour[0] = start_node
    current = start_node

    for step in range(1, n):
        dists = dist_matrix[current]
        unvis_idx = np.where(unvisited)[0]
        next_city = unvis_idx[np.argmin(dists[unvis_idx])]
        tour[step] = next_city
        unvisited[next_city] = False
        current = next_city

    return tour


def nearest_neighbor_generator(dist_matrix, start_node=0, frame_interval=5):
    """
    Step-by-step generator for animating Nearest Neighbor.
    Yields (current_subtour, current_cost, progress_percentage, description).
    """
    n = dist_matrix.shape[0]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start_node] = False
    tour = [start_node]
    current = start_node

    yield (list(tour), 0.0, 0.0, f"Starting Nearest Neighbor from City {start_node+1}")

    for step in range(1, n):
        dists = dist_matrix[current]
        unvis_idx = np.where(unvisited)[0]
        next_city = int(unvis_idx[np.argmin(dists[unvis_idx])])
        tour.append(next_city)
        unvisited[next_city] = False
        current = next_city

        if step % frame_interval == 0 or step == n - 1:
            # Partial path distance
            sub_arr = np.array(tour, dtype=np.int32)
            cost = float(np.sum(dist_matrix[sub_arr[:-1], sub_arr[1:]]))
            if step == n - 1:
                cost += dist_matrix[tour[-1], tour[0]]
            progress = (step + 1) / n
            yield (list(tour), cost, progress, f"Added City {next_city+1} ({step+1}/{n})")


def multi_start_nearest_neighbor(dist_matrix, max_starts=20):
    """Runs Nearest Neighbor across multiple evenly spaced starts."""
    n = dist_matrix.shape[0]
    starts = np.linspace(0, n - 1, min(n, max_starts), dtype=int)
    best_tour = None
    best_cost = float("inf")

    for s in starts:
        tour = nearest_neighbor(dist_matrix, start_node=int(s))
        cost = calculate_tour_length(tour, dist_matrix)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour


# -------------------------------------------------------------
# 2. Local Search Operators
# -------------------------------------------------------------

def two_opt(tour, dist_matrix, max_passes=50):
    """Vectorized / fast 2-opt local search (First-Improvement)."""
    tour = np.array(tour, dtype=np.int32).copy()
    n = len(tour)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 2):
            a, b = tour[i], tour[i + 1]
            d_ab = dist_matrix[a, b]

            max_j = n if i > 0 else n - 1
            j_indices = np.arange(i + 2, max_j)
            c = tour[j_indices]
            d = tour[(j_indices + 1) % n]

            deltas = (dist_matrix[a, c] + dist_matrix[b, d]) - (d_ab + dist_matrix[c, d])
            best_idx = np.where(deltas < -1e-6)[0]

            if len(best_idx) > 0:
                j = j_indices[best_idx[0]]
                tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                improved = True
                break
        if not improved:
            break

    return tour


def two_opt_generator(initial_tour, dist_matrix, max_moves=100):
    """
    Step-by-step generator for animating 2-opt untangling edge moves.
    Yields (tour, cost, improvement_delta, move_description).
    """
    tour = np.array(initial_tour, dtype=np.int32).copy()
    n = len(tour)
    cost = calculate_tour_length(tour, dist_matrix)
    yield (tour.copy(), cost, 0.0, "Initial Tour before 2-Opt")

    moves_count = 0
    improved = True

    while improved and moves_count < max_moves:
        improved = False
        for i in range(n - 2):
            a, b = tour[i], tour[i + 1]
            d_ab = dist_matrix[a, b]

            max_j = n if i > 0 else n - 1
            j_indices = np.arange(i + 2, max_j)
            c = tour[j_indices]
            d = tour[(j_indices + 1) % n]

            deltas = (dist_matrix[a, c] + dist_matrix[b, d]) - (d_ab + dist_matrix[c, d])
            best_idx = np.where(deltas < -1e-6)[0]

            if len(best_idx) > 0:
                j = j_indices[best_idx[0]]
                delta = deltas[best_idx[0]]
                tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                cost += delta
                moves_count += 1
                improved = True
                yield (tour.copy(), cost, delta, f"2-Opt Move #{moves_count}: Untangled edges ({a+1}-{b+1}) & ({tour[j]+1}-{d+1}), saved {-delta:.1f} km")
                break


def node_relocation(tour, dist_matrix, max_passes=20):
    """1-opt / Node insertion move: removes node tour[i] and re-inserts between (u, v)."""
    tour = list(tour)

    for _ in range(max_passes):
        improved = False
        for i in range(len(tour)):
            node = tour[i]
            prev_i = tour[i - 1]
            next_i = tour[(i + 1) % len(tour)]

            cost_remove = dist_matrix[prev_i, node] + dist_matrix[node, next_i] - dist_matrix[prev_i, next_i]

            best_saving = 0
            best_j = -1
            for j in range(len(tour)):
                if j == i or j == (i - 1) % len(tour):
                    continue
                u = tour[j]
                v = tour[(j + 1) % len(tour)]
                cost_add = dist_matrix[u, node] + dist_matrix[node, v] - dist_matrix[u, v]
                saving = cost_remove - cost_add
                if saving > 1e-6 and saving > best_saving:
                    best_saving = saving
                    best_j = j

            if best_j != -1:
                tour.pop(i)
                if best_j > i:
                    best_j -= 1
                tour.insert(best_j + 1, node)
                improved = True
                break
        if not improved:
            break

    return np.array(tour, dtype=np.int32)


# -------------------------------------------------------------
# 3. Metaheuristics: Iterated Local Search (ILS) with Double-Bridge
# -------------------------------------------------------------

def double_bridge_perturbation(tour):
    """
    Double bridge (4-opt) perturbation.
    Cuts the tour into 4 pieces and reassembles them in non-adjacent order (A-D-C-B).
    """
    n = len(tour)
    if n < 8:
        t = tour.copy()
        np.random.shuffle(t)
        return t

    pos = sorted(random.sample(range(1, n), 3))
    i, j, k = pos[0], pos[1], pos[2]

    return np.concatenate([
        tour[:i],
        tour[k:],
        tour[j:k],
        tour[i:j]
    ])


def iterated_local_search_generator(dist_matrix, initial_tour=None, max_iterations=40):
    """
    Step-by-step generator for animating Iterated Local Search with Double Bridge kicks.
    Yields (tour, cost, is_new_global_best, step_description).
    """
    n = dist_matrix.shape[0]
    if initial_tour is None:
        initial_tour = nearest_neighbor(dist_matrix, 0)

    current_tour = two_opt(initial_tour, dist_matrix)
    current_cost = calculate_tour_length(current_tour, dist_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost

    yield (best_tour.copy(), best_cost, True, f"Starting ILS after Initial 2-Opt: {best_cost:,.0f} km")

    for it in range(1, max_iterations + 1):
        # 4-Opt Double Bridge Kick
        perturbed = double_bridge_perturbation(current_tour)
        # Local search
        candidate = two_opt(perturbed, dist_matrix, max_passes=25)
        cand_cost = calculate_tour_length(candidate, dist_matrix)

        if cand_cost < current_cost:
            current_tour = candidate
            current_cost = cand_cost
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_tour = candidate.copy()
                yield (best_tour.copy(), best_cost, True, f"Iteration {it}: 🎉 New Best Solution Found! {best_cost:,.0f} km")
            else:
                yield (current_tour.copy(), current_cost, False, f"Iteration {it}: Accepted improving move ({current_cost:,.0f} km)")
        else:
            temp = max(1.0, best_cost * 0.005)
            if random.random() < math.exp((current_cost - cand_cost) / temp):
                current_tour = candidate
                current_cost = cand_cost
                yield (current_tour.copy(), current_cost, False, f"Iteration {it}: Accepted uphill kick to escape local minimum")


def iterated_local_search(
    dist_matrix,
    initial_tour=None,
    max_time_sec=8.0,
    progress_callback=None
):
    """Iterated Local Search (ILS) batch execution."""
    start_time = time.time()

    if initial_tour is None:
        initial_tour = nearest_neighbor(dist_matrix, 0)

    current_tour = two_opt(initial_tour, dist_matrix)
    current_cost = calculate_tour_length(current_tour, dist_matrix)

    best_tour = current_tour.copy()
    best_cost = current_cost

    history = [(0.0, best_cost)]
    iteration = 0

    while (time.time() - start_time) < max_time_sec:
        iteration += 1
        perturbed = double_bridge_perturbation(current_tour)
        candidate_tour = two_opt(perturbed, dist_matrix, max_passes=25)
        candidate_cost = calculate_tour_length(candidate_tour, dist_matrix)

        if candidate_cost < current_cost:
            current_tour = candidate_tour
            current_cost = candidate_cost

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_tour = candidate_tour.copy()
                elapsed = time.time() - start_time
                history.append((elapsed, best_cost))
                if progress_callback:
                    progress_callback(iteration, best_cost, best_tour)
        else:
            temp = max(1.0, best_cost * 0.005)
            if random.random() < math.exp((current_cost - candidate_cost) / temp):
                current_tour = candidate_tour
                current_cost = candidate_cost

    return best_tour, best_cost, history


def simulated_annealing(
    dist_matrix,
    initial_tour=None,
    initial_temp=100.0,
    cooling_rate=0.995,
    max_iterations=10000,
    progress_callback=None
):
    """Simulated Annealing heuristic with 2-opt neighborhood swap."""
    n = dist_matrix.shape[0]
    if initial_tour is None:
        initial_tour = nearest_neighbor(dist_matrix, 0)

    current_tour = initial_tour.copy()
    current_cost = calculate_tour_length(current_tour, dist_matrix)
    best_tour = current_tour.copy()
    best_cost = current_cost

    temp = initial_temp
    history = [(0, best_cost)]

    for it in range(1, max_iterations + 1):
        i, j = sorted(random.sample(range(n), 2))
        if j == (i + 1) or (i == 0 and j == n - 1):
            continue

        a, b = current_tour[i], current_tour[(i + 1) % n]
        c, d = current_tour[j], current_tour[(j + 1) % n]

        delta = (dist_matrix[a, c] + dist_matrix[b, d]) - (dist_matrix[a, b] + dist_matrix[c, d])

        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-8)):
            current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
                history.append((it, best_cost))
                if progress_callback and it % 100 == 0:
                    progress_callback(it, best_cost, best_tour)

        temp *= cooling_rate
        if temp < 1e-4:
            break

    return best_tour, best_cost, history
