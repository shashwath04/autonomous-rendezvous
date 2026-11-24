import math
import random
import numpy as np

class Node():
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

# Find the distance between two points
def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

# Find the angle needed to step in to reach the goal state and step towards it
def steer(x0, xf, step_size):
    start = np.asarray(x0, dtype=float)
    target = np.asarray(xf, dtype=float)
    delta = target - start
    dist = np.linalg.norm(delta)

    if dist <= step_size:
        return target
    return start + (step_size / dist) * delta

# Check if there are any obstacles between points a and b
def is_segment_free(a, b, obstacles, step = 0.1):
    dist = distance(a, b)
    n_checks = max(1, int(math.ceil(dist / step)))
    for i in range(n_checks + 1):
        t = i / n_checks
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        for ox, oy, r in obstacles:
            if np.sqrt((x - ox) ** 2 + (y - ox) ** 2) <= r:
                return False
    return True

def plan_rrt(start, goal, bounds, obstacles, step_size=0.5, goal_bias=0.08, goal_tol=0.4, max_iter=3000, seed=42):
    rng = random.Random(seed)
    nodes = [Node(start)]
    (x_min, x_max), (y_min, y_max) = bounds

    for _ in range(max_iter):

        # 8% (or goal_bias) chance that the sample we are trying to get to is just the goal, 92% (1 - goal_bias)
        # chance that we are just trying to get to any random point that is within bounds.
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))

        # Find the nearest node to the the sample and steer towards it
        nearest = min(nodes, key=lambda n: distance(n.state, sample))
        new_state = steer(nearest.state, sample, step_size)

        # Check for out of bounds and if there are obstacles between
        if not (x_min <= new_state[0] <= x_max and y_min <= new_state[1] <= y_max):
            continue
        if not is_segment_free(nearest.state, new_state, obstacles):
            continue

        # Establish the new node as a class Node and add it to the list of nodes.
        new_node = Node(new_state, parent=nearest)
        nodes.append(new_node)

        # If we reached the goal, trace the path up to the parent and return it, as well as the list of nodes.
        if distance(new_state, goal) <= goal_tol and is_segment_free(new_state, goal, obstacles):
            goal_node = Node(goal, parent=new_node)
            path = []
            curr = goal_node
            while curr is not None:
                path.append(curr.state)
                curr = curr.parent
            return path[::-1], nodes

    return None, nodes

def plan_rrt_star(start, goal, bounds, obstacles, step_size=0.5, goal_bias=0.08, goal_tol=0.4, max_iter=3000, seed=42):
    return