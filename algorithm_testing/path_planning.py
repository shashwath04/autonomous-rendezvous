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

def hypersphere(tree, x_new, radius):
    nodes = []

    for node in tree:
        if distance(node.state, x_new) <= radius:
            nodes.append(node)

    return nodes

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

class NodeStar:
    def __init__(self, state, cost=np.inf, parent=None):
        self.state = state
        self.cost = cost
        self.parent = parent
        self.children = []


def rewire(node, new_parent, new_cost):
    node.parent.children.remove(node)
    node.parent = new_parent
    new_parent.children.append(node)
    node.cost = new_cost

    stack = [node]
    while stack:
        parent = stack.pop()
        for child in parent.children:
            child.cost = parent.cost + distance(parent.state, child.state)
            stack.append(child)


def plan_rrt_star(
    start, goal, bounds, radius, obstacles,
    step_size=0.5, goal_bias=0.08, goal_tol=0.4,
    max_iter=3000, seed=42,
):
    rng = random.Random(seed)
    root = NodeStar(start, cost=0.0)
    nodes = [root]
    goal_candidates = []  
    (x_min, x_max), (y_min, y_max) = bounds
    eps = 1e-9

    def consider_goal(node):
        goal_length = distance(node.state, goal)
        if goal_length <= goal_tol and is_segment_free(
            node.state, goal, obstacles
        ):
            goal_candidates.append((node, goal_length))

    consider_goal(root)

    for _ in range(max_iter):
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(x_min, x_max), rng.uniform(y_min, y_max))

        nearest = min(nodes, key=lambda node: distance(node.state, sample))
        new_state = steer(nearest.state, sample, step_size)
        nearest_length = distance(nearest.state, new_state)

        if nearest_length <= eps:
            continue
        if not (x_min <= new_state[0] <= x_max and y_min <= new_state[1] <= y_max):
            continue
        if not is_segment_free(nearest.state, new_state, obstacles):
            continue

        near_nodes = hypersphere(nodes, new_state, radius)

    
        best_parent = nearest
        best_cost = nearest.cost + nearest_length

        for neighbor in near_nodes:
            edge_length = distance(neighbor.state, new_state)
            candidate_cost = neighbor.cost + edge_length
            if candidate_cost >= best_cost - eps:
                continue
            if is_segment_free(neighbor.state, new_state, obstacles):
                best_parent = neighbor
                best_cost = candidate_cost

        new_node = NodeStar(new_state, cost=best_cost, parent=best_parent)
        best_parent.children.append(new_node)
        nodes.append(new_node)

        # Rewire existing neighbors if going through new_node is cheaper.
        for neighbor in near_nodes:
            if neighbor is root:
                continue
            edge_length = distance(new_node.state, neighbor.state)
            candidate_cost = new_node.cost + edge_length
            if candidate_cost >= neighbor.cost - eps:
                continue
            if is_segment_free(new_node.state, neighbor.state, obstacles):
                rewire(neighbor, new_node, candidate_cost)

        consider_goal(new_node)

    if not goal_candidates:
        return None, nodes

    # Read costs after all rewiring. An earlier candidate may now be cheaper.
    best_node, goal_length = min(
        goal_candidates, key=lambda entry: entry[0].cost + entry[1]
    )

    path = [goal] if goal_length > 0.0 else []
    current = best_node
    while current is not None:
        path.append(current.state)
        current = current.parent

    return path[::-1], nodes