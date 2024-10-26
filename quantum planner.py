from pyqubo import Binary
import numpy as np
import re
import matplotlib.pyplot as plt
import gurobi_optimods as go

n=2
plane_range = list(range(2+1))
airport_range = list(range(1,n+1))
passenger_range = list(range(2))
time_range = list(range(5))

distances = np.array([[0,4,4],[4,0,4],[4,4,0]])
distances

passenger_destinations = np.array([1,2])
passenger_start = np.array([2,1])
airplane_start = np.array([2,1])

def set_conditions(P, L):
    #At the start the only possition of the passenger is at start airport (isn't at the others)
    for r in passenger_range:
        for i in airport_range:
            if i != passenger_start[r]:
                L[r, time_range[0], i, :] = 0

    #At the start the only possition of the airplane is at start airport
    for p in plane_range[:-1]:
        for i in airport_range:
            P[time_range[0], i, p] = (airplane_start[p] == i)
    
    #At the end the only available possition of the passenger is in airport
    for r in passenger_range: 
        for i in airport_range:
            for p in plane_range:
                if i != passenger_destinations[r]:
                    L[r, time_range[-1], i, p] = 0

    #Put a plane that stays at all airports at the same time
    for t in time_range:
        for i in airport_range:
            P[t, i, plane_range[-1]] = 1

def constrain_function(P, L, verbose=False):
    
    only_one_state = np.sum((np.sum(P, axis=1) - 1)**2)
    if verbose:
        print("only_one,state",only_one_state)
    
    _to_sum = []
    for t1 in time_range:
        for t2 in time_range:
            for i in airport_range:
                for j in airport_range:
                    if i!=j and abs(t1-t2) < distances[i-1,j-1]:
                        _to_sum.append(P[t1,i,:]*P[t2,j,:])
    cond_P1 = np.sum(_to_sum)
    if verbose:
        print("cond_P1", cond_P1, end=" ")

    _to_sum = []
    for t in time_range[:-1]:
        _to_sum.append(np.array(P[t,1:n+1,:]*P[t+1,0:1,:]))
    cond_P2 = np.sum(_to_sum)
    if verbose:
        print("cond_P2", cond_P2, end=" ")

    _to_sum = []
    for t in time_range[:-1]:
        _to_sum.append(np.array(P[t,n+1,:]*P[t+1,1:n+2,:]))
    cond_P3 = np.sum(_to_sum)
    if verbose:
        print("cond_P3", cond_P3, end=" ")

    _to_sum = []
    for t in time_range[:-1]:
        _to_sum.append(np.array(P[t,0,:]*P[t+1,n+1,:]))
    cond_P4 = np.sum(_to_sum)
    if verbose:
        print("cond_P4", cond_P4)

    conditions_on_P = only_one_state + cond_P1 + cond_P2 + cond_P3 + cond_P4



    only_one_place = np.sum((np.sum(np.sum(10*L, axis=-1), axis=-1) - 10)**2)
    if verbose:
        print("only_one_place", only_one_place)
    
    _to_sum = []
    for r in passenger_range:
        cond_dest_reached = _to_sum.append((np.sum(2*L[r, time_range[-1], passenger_destinations[r], :],  axis=-1)-2)**2 )
    cond_dest_reached = np.sum(_to_sum)
    if verbose:
        print("was destination reached ",cond_dest_reached)
        
    #P = np.expand_dims(P, 0)

    _to_sum = []
    for r in passenger_range:
        for i in set([0,n+1]) | set(airport_range):# - set([passenger_destinations[r]]):
            _to_sum.append(np.sum(-L[r,:,i,:]*P[:,i,:]))
    cond1 = np.sum(_to_sum)
    if verbose:
        print("in_plane", cond1, end=" ")

    _to_sum = []
    for t in time_range[:-1]:
        for i in [0] + airport_range:
            _to_sum.append(np.sum(-L[:,t,0,:]*L[:,t+1,i,:]))
    cond2 = np.sum(_to_sum)
    if verbose:
        print("from_air", cond2, end=" ")

    _to_sum = []
    for r in passenger_range:
        for t in time_range[:-1]:
            for u in plane_range:
                for k in plane_range:
                    for i in set(airport_range) - set([passenger_destinations[r]]):
                        _to_sum.append(np.sum(-L[r, t, i, u] * L[r, t+1, i, k]))
    cond3 = np.sum(_to_sum)
    if verbose:
        print("port_to_port", cond3, end=" ")

    _to_sum = []
    for r in passenger_range:
        for t in time_range[:-1]:
            for i in set(airport_range) - set([passenger_destinations[r]]):
                _to_sum.append(np.sum(-L[r, t, i, :] * L[r, t+1, n+1, :]))
    cond4 = np.sum(_to_sum)
    if verbose:
        print("port_to_start", cond4, end=" ")

    _to_sum = []
    for t in time_range[:-1]:
        _to_sum.append(np.sum(-L[:, t, n+1, :] * L[:, t+1, 0, :]))
    cond5 = np.sum(_to_sum)
    if verbose:
        print("start_to_air", cond5, end=" ")

    _to_sum = []
    for r in passenger_range:
        _x = passenger_destinations[r]
        for t in time_range[:-1]:
            for u in plane_range:
                for k in plane_range:
                    _to_sum.append(np.sum(-L[r, t, _x, u] * L[r, t+1, _x, k]))
    cond6 = np.sum(_to_sum)
    if verbose:
        print("dest_to_dest", cond6)

    if verbose:
        print("cond1-6", cond1 + cond2 + cond3 + cond4 + cond5 + cond6)

    conditions_on_LP = only_one_place + cond1 + cond2 + \
                    cond3 + cond4 + cond5 + cond6 + cond_dest_reached
    return conditions_on_LP + conditions_on_P

def generate_binary_variables():
    P = []
    for t in time_range:
        cur1 = []
        for a in [0] + airport_range + [n+1]:
            cur1.append([Binary(f"P_t{t},A{a},B{b}") for b in plane_range])
        P.append(cur1)
    P = np.array(P)
    P.shape

    L = []
    for r in passenger_range:
        cur1 = []
        for t in time_range:
            cur2 = []
            for i in [0] + airport_range + [n+1]:
                cur3 = []
                for k in plane_range:
                    cur3.append(Binary(f"L_C{r},t{t},A{i},B{k}"))
                cur2.append(cur3)
            cur1.append(cur2)
        L.append(cur1)
    L = np.array(L)
    return P, L

def model_to_matrix(model):
    qubo_dict, offset = model.to_qubo()

    # Step 3: Create a list of unique variables
    variables = sorted(set(i for pair in qubo_dict.keys() for i in pair))
    print(f"There are {len(variables)} variables in the model")
    index_map = {var: idx for idx, var in enumerate(variables)}

    # Initialize the QUBO matrix with zeros
    matrix_size = len(variables)
    qubo_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the matrix using the QUBO dictionary
    for (i, j), value in qubo_dict.items():
        idx_i = index_map[i]
        idx_j = index_map[j]
        qubo_matrix[idx_i, idx_j] = value
    return qubo_matrix, variables


def to_matrix_result(data):
    L_res = np.zeros([len(passenger_range), 
                  len(time_range),
                  len(airport_range)+2,
                  len(plane_range)])
    P_res = np.zeros([len(time_range),
                  len(airport_range)+2,
                  len(plane_range)])

    set_conditions(P_res, L_res)

    for key, val in  data:
        if key[0] == "P":
            time, state, plane = re.match("P_t(\d*),A(\d*),B(\d*)", key).groups()
            time, state, plane = int(time), int(state), int(plane)
            P_res[time, state, plane] = val
        if key[0] == "L":
            passenger, time, state, plane = re.match("L_C(\d*),t(\d*),A(\d*),B(\d*)", key).groups()
            passenger, time, state, plane = int(passenger), int(time), int(state), int(plane)
            L_res[passenger, time, state, plane] = val
    return P_res, L_res

def show_result(P_res, L_res, figsize):
    fig, ax = plt.subplots(1,3, figsize=figsize)
    plane_line = [[] for p in plane_range[:-1]]
    positions=[]
    for plane in plane_range[:-1]:
        for t in time_range:
            for state in [0] + airport_range + [n+1]:
                if P_res[t,state,plane]:
                    plane_line[plane].append((t,state))
                    positions.append((t,state,plane) )

    positions_pass = []
    for r in passenger_range:
        positions_pass.append([])
        for t in time_range:
            for state in [0] + airport_range + [n+1]:
                for plane in plane_range[:-1]:
                    if L_res[r,t,state,plane]:
                        positions_pass[r].append((t,state, plane) )
                        
    positions = np.array(sorted(positions, key = lambda x: x[0]))
    ax[0].scatter(positions[:,0], positions[:,1]+positions[:,2]/5, c=positions[:,2],
                  vmin=0, vmax=len(plane_range)-1, alpha=0.5)
    ax[0].set_title("plane positions")
    for r in passenger_range:
        positions1 = np.array(positions_pass[r])
        ax[r+1].scatter(positions1[:,0], positions1[:,1]+positions1[:,2]/5, c=positions1[:,2],
                        vmin=0, vmax=len(plane_range)-1, alpha=0.5)
        ax[r+1].hlines(y=passenger_destinations[r], xmin=0, xmax=np.max(positions1[:,0]), color="black")
        for i, lines in enumerate(plane_line):
            lines = np.array(sorted(lines, key = lambda x: x[0]))
            ax[r+1].plot(lines[:,0], lines[:,1]+i/5, c=["red", "green"][i], alpha=0.5)
        ax[r+1].set_title(f"passenger {r}")


P,L = generate_binary_variables()
set_conditions(P, L)

model = constrain_function(P,L).compile()

qubo_matrix, variables = model_to_matrix(model)

from gurobi_optimods.qubo import solve_qubo
result = solve_qubo(qubo_matrix)

print("Solution to the QUBO problem:", result)
print("\noptimal")
P_res_gurobi, L_res_gurobi = to_matrix_result( zip(variables, np.array(result.solution)))
print(constrain_function(P_res_gurobi, L_res_gurobi, verbose=True))
show_result(P_res_gurobi, L_res_gurobi, figsize=(9,3))
plt.show()