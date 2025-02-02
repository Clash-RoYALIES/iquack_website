import itertools
from problem_dimensions import N, T, alpha, flow, d

def cost_flow(perm, f):
    """
    Compute the flow cost for a given assignment 'perm' (a tuple where perm[j] is the location
    assigned to facility j) and a flow matrix f:
    """
    return sum(f[j][k] * d[perm[j]][perm[k]] for j in range(N) for k in range(N))

def cost_move(perm_old, perm):
    """
    Compute the movement cost for moving from assignment 'perm_old' to 'perm':
    """
    return sum(alpha * d[perm_old[j]][perm[j]] for j in range(N))

results = []

# For the first month, assume the initial assignment is given by all facilities at location 0.
perm_old = (0,) * N

print("Classical Brute-Force Time-Dependent QAP Solution:")
overall_total_cost = 0

for month in range(1, T + 1):
    print(f"\nMonth {month}:")
    f_current = flow[month - 1]
    
    best_total_cost = float('inf')
    best_perm = None

    for perm in itertools.permutations(range(N)):
        flow_cost_val = cost_flow(perm, f_current)
        move_cost_val = cost_move(perm_old, perm)
        total_cost = flow_cost_val + move_cost_val
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_perm = perm

    print("  Best assignment (facility j -> location):", best_perm)
    flow_cost_val = cost_flow(best_perm, f_current)
    move_cost_val = cost_move(perm_old, best_perm)
    print("  Flow cost:", flow_cost_val)
    print("  Movement cost:", move_cost_val)
    print("  Total cost for Month", month, ":", best_total_cost)
    
    overall_total_cost += best_total_cost
    results.append((month, best_total_cost, best_perm))
    
    perm_old = best_perm

print(f"\nOverall Total Cost over all {T} months: {overall_total_cost}")