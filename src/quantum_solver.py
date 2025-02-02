from dimod import ConstrainedQuadraticModel, Binary
from dwave.system import LeapHybridCQMSampler
from problem_dimensions import N, T, alpha, flow, d
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

cqm = ConstrainedQuadraticModel()

# x[(j, m, t)] = 1 if facility j is assigned to location m at time t.
x = {(j, m, t): Binary(f'x_{j}_{m}_{t}')
     for j in range(N) for m in range(N) for t in range(T)}

# (1) Each facility is assigned exactly one location per time step.
for t in range(T):
    for j in range(N):
        cqm.add_constraint(sum(x[(j, m, t)] for m in range(N)) == 1, label=f'facility_{j}_at_time_{t}')
        
# (2) Each location gets exactly one facility per time step.
for t in range(T):
    for m in range(N):
        cqm.add_constraint(sum(x[(j, m, t)] for j in range(N)) == 1, label=f'location_{m}_at_time_{t}')

# (A) Flow cost for each time step:
flow_cost_per_month = [0] * T
for t in range(T):
    f_t = flow[t]  # Flow matrix at time t
    for j in range(N):
        for k in range(N):
            for m in range(N):
                for n in range(N):
                    flow_cost_per_month[t] += f_t[j][k] * d[m][n] * x[(j, m, t)] * x[(k, n, t)]

# (B) Movement cost between consecutive time steps:
move_cost_per_month = [0] * (T - 1)
for t in range(T-1):
    for j in range(N):
        for m in range(N):
            for n in range(N):
                move_cost_per_month[t] += alpha * d[m][n] * x[(j, m, t)] * x[(j, n, t+1)]

cqm.set_objective(sum(flow_cost_per_month) + sum(move_cost_per_month))

sampler = LeapHybridCQMSampler()

solution = sampler.sample_cqm(cqm)

feasible_solutions = solution.filter(lambda d: d.is_feasible)
if len(feasible_solutions):
    best_solution = feasible_solutions.first.sample

    print("Optimal facility-to-location assignments over time:")
    for t in range(T):
        print(f"\nTime step {t}:")
        for j in range(N):
            for m in range(N):
                if best_solution[f'x_{j}_{m}_{t}'] == 1:
                    print(f"  Facility {j} -> Location {m}")

    total_flow_cost = 0
    total_move_cost = 0

    for t in range(T):
        flow_cost_value = 0
        if t < len(flow_cost_per_month):
            for j in range(N):
                for k in range(N):
                    for m in range(N):
                        for n in range(N):
                            flow_cost_value += flow[t][j][k] * d[m][n] * best_solution[f'x_{j}_{m}_{t}'] * best_solution[f'x_{k}_{n}_{t}']
        total_flow_cost += flow_cost_value

        move_cost_value = 0
        if t < len(move_cost_per_month):
            for j in range(N):
                for m in range(N):
                    for n in range(N):
                        move_cost_value += alpha * d[m][n] * best_solution[f'x_{j}_{m}_{t}'] * best_solution[f'x_{j}_{n}_{t+1}']
        total_move_cost += move_cost_value


    print(f"\nTotal Flow Cost for the Year: {total_flow_cost}")
    print(f"Total Movement Cost for the Year: {total_move_cost}")
    print(f"Total Cost for the Year: {total_flow_cost + total_move_cost}")

else:
    print("No feasible solution found.")


assignments = {}
for t in range(T):
    assignments[t] = {}
    for j in range(N):
        for m in range(N):
            if best_solution[f'x_{j}_{m}_{t}'] == 1:
                assignments[t][j] = m

flow_cost_per_month = [0] * T
for t in range(T):
    for j in range(N):
        for k in range(N):
            for m in range(N):
                for n in range(N):
                    flow_cost_per_month[t] += flow[t][j][k] * d[m][n] * best_solution[f'x_{j}_{m}_{t}'] * best_solution[f'x_{k}_{n}_{t}']

move_cost_per_month = [0] * (T - 1)
for t in range(T-1):
    for j in range(N):
        for m in range(N):
            for n in range(N):
                move_cost_per_month[t] += alpha * d[m][n] * best_solution[f'x_{j}_{m}_{t}'] * best_solution[f'x_{j}_{n}_{t+1}']

colors = plt.cm.get_cmap('tab20', N).colors

plt.figure(figsize=(14, 10))

# Plot 1: Facility locations over time with movement paths
plt.subplot(2, 1, 1)
for j in range(N):
    x = list(range(T))
    y = [assignments[t][j] for t in range(T)]
    plt.plot(x, y, marker='o', linestyle='-', color=colors[j], label=f'Facility {j}')

for t in range(T-1):
    plt.annotate(f'{move_cost_per_month[t]:.0f}', 
                 xy=(t+0.5, (N-1)*1.05), 
                 ha='center', va='bottom',
                 fontsize=8, color='red')

plt.title('Facility Locations and Movement Paths Over Time')
plt.xlabel('Time Step')
plt.ylabel('Location')
plt.xticks(range(T))
plt.yticks(range(N))
plt.grid(True)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()