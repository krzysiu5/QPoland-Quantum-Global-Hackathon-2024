from pyqubo import Binary
import numpy as np
import re
import matplotlib.pyplot as plt
import gurobi_optimods as go


class QuantumPlanner:
    def __init__(self, number_of_planes, number_of_airport, number_of_passengers, number_of_time_periods,
                      distances, passenger_destinations, passenger_start, airplane_start):
        self.n = number_of_airport
        self.plane_range = list(range(number_of_planes+1))
        self.airport_range = list(range(1,self.n+1))
        self.passenger_range = list(range(number_of_passengers))
        self.time_range = list(range(number_of_time_periods))

        self.distances = distances 

        self.passenger_destinations = passenger_destinations
        self.passenger_start = passenger_start
        self.airplane_start = airplane_start

    def set_conditions(self, P, L):
        #At the start the only possition of the passenger is at start airport (isn't at the others)
        for r in self.passenger_range:
            for i in self.airport_range + [0, self.n+1]:
                if i != self.passenger_start[r]:
                    L[r, self.time_range[0], i, :] = 0

        #At the start the only possition of the airplane is at start airport
        for p in self.plane_range[:-1]:
            for i in self.airport_range + [0,self.n+1]:
                P[self.time_range[0], i, p] = (self.airplane_start[p] == i)
        
        #At the end the only available possition of the passenger is in airport
        for r in self.passenger_range: 
            for i in self.airport_range + [0, self.n+1]:
                for p in self.plane_range:
                    if i != self.passenger_destinations[r]:
                        L[r, self.time_range[-1], i, p] = 0

        #Put a plane that stays at all airports at the same time
        for t in self.time_range:
            for i in self.airport_range:
                P[t, i, self.plane_range[-1]] = 1
            for i in [0, self.n+1]: # the plane is never in the air or taking-off
                P[t, i, self.plane_range[-1]] = 0

    def constrain_function(self, P, L, verbose=False):
        
        only_one_state = np.sum((np.sum(P, axis=1) - 1)**2)
        if verbose:
            print("only_one,state",only_one_state)
        
        _to_sum = []
        for t1 in self.time_range:
            for t2 in self.time_range:
                for i in self.airport_range:
                    for j in self.airport_range:
                        if i!=j and abs(t1-t2) < self.distances[i-1,j-1]:
                            _to_sum.append(P[t1,i,:]*P[t2,j,:])
        cond_P1 = np.sum(_to_sum)
        if verbose:
            print("cond_P1", cond_P1, end=" ")

        _to_sum = []
        for t in self.time_range[:-1]:
            _to_sum.append(np.array(P[t,1:self.n+1,:]*P[t+1,0:1,:]))
        cond_P2 = np.sum(_to_sum)
        if verbose:
            print("cond_P2", cond_P2, end=" ")

        _to_sum = []
        for t in self.time_range[:-1]:
            _to_sum.append(np.array(P[t,self.n+1,:]*P[t+1,1:self.n+2,:]))
        cond_P3 = np.sum(_to_sum)
        if verbose:
            print("cond_P3", cond_P3, end=" ")

        _to_sum = []
        for t in self.time_range[:-1]:
            _to_sum.append(np.array(P[t,0,:]*P[t+1,self.n+1,:]))
        cond_P4 = np.sum(_to_sum)
        if verbose:
            print("cond_P4", cond_P4)

        conditions_on_P = only_one_state + cond_P1 + cond_P2 + cond_P3 + cond_P4



        only_one_place = np.sum((np.sum(np.sum(10*L, axis=-1), axis=-1) - 10)**2)
        if verbose:
            print("only_one_place", only_one_place)
        
        _to_sum = []
        for r in self.passenger_range:
            for i in set([0,self.n+1]) | set(self.airport_range):
                _to_sum.append(np.sum(-L[r,:,i,:]*P[:,i,:]))
        cond1 = np.sum(_to_sum)
        if verbose:
            print("in_plane", cond1, end=" ")

        _to_sum = []
        for t in self.time_range[:-1]:
            for i in [0] + self.airport_range:
                _to_sum.append(np.sum(-L[:,t,0,:]*L[:,t+1,i,:]))
        cond2 = np.sum(_to_sum)
        if verbose:
            print("from_air", cond2, end=" ")

        _to_sum = []
        for r in self.passenger_range:
            for t in self.time_range[:-1]:
                for u in self.plane_range:
                    for k in self.plane_range:
                        for i in set(self.airport_range) - set([self.passenger_destinations[r]]):
                            _to_sum.append(np.sum(-L[r, t, i, u] * L[r, t+1, i, k]))
        cond3 = np.sum(_to_sum)
        if verbose:
            print("port_to_port", cond3, end=" ")

        _to_sum = []
        for r in self.passenger_range:
            for t in self.time_range[:-1]:
                for i in set(self.airport_range) - set([self.passenger_destinations[r]]):
                    _to_sum.append(np.sum(-L[r, t, i, :] * L[r, t+1, self.n+1, :]))
        cond4 = np.sum(_to_sum)
        if verbose:
            print("port_to_start", cond4, end=" ")

        _to_sum = []
        for t in self.time_range[:-1]:
            _to_sum.append(np.sum(-L[:, t, self.n+1, :] * L[:, t+1, 0, :]))
        cond5 = np.sum(_to_sum)
        if verbose:
            print("start_to_air", cond5, end=" ")

        _to_sum = []
        for r in self.passenger_range:
            _x = self.passenger_destinations[r]
            for t in self.time_range[:-1]:
                for u in self.plane_range:
                    for k in self.plane_range:
                        _to_sum.append(np.sum(-L[r, t, _x, u] * L[r, t+1, _x, k]))
        cond6 = np.sum(_to_sum)
        if verbose:
            print("dest_to_dest", cond6)

        if verbose:
            print("cond1-6", cond1 + cond2 + cond3 + cond4 + cond5 + cond6)

        conditions_on_LP = only_one_place + cond1 + cond2 + \
                        cond3 + cond4 + cond5 + cond6
        return conditions_on_LP + conditions_on_P

    def generate_binary_variables(self):
        P = []
        for t in self.time_range:
            cur1 = []
            for a in [0] + self.airport_range + [self.n+1]:
                cur1.append([Binary(f"P_t{t},A{a},B{b}") for b in self.plane_range])
            P.append(cur1)
        P = np.array(P)
        P.shape

        L = []
        for r in self.passenger_range:
            cur1 = []
            for t in self.time_range:
                cur2 = []
                for i in [0] + self.airport_range + [self.n+1]:
                    cur3 = []
                    for k in self.plane_range:
                        cur3.append(Binary(f"L_C{r},t{t},A{i},B{k}"))
                    cur2.append(cur3)
                cur1.append(cur2)
            L.append(cur1)
        L = np.array(L)
        return P, L

    def model_to_matrix(self, model):
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


    def to_matrix_result(self, data, additional_conditions=None):
        L_res = np.zeros([len(self.passenger_range), 
                    len(self.time_range),
                    len(self.airport_range)+2,
                    len(self.plane_range)])
        P_res = np.zeros([len(self.time_range),
                    len(self.airport_range)+2,
                    len(self.plane_range)])

        self.set_conditions(P_res, L_res)
        if additional_conditions is not None:
            additional_conditions(self, P_res, L_res)

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

    def show_result(self, P_res, L_res, figsize):
        fig, ax = plt.subplots(1,3, figsize=figsize)
        plane_line = [[] for p in self.plane_range[:-1]]
        positions=[]
        for plane in self.plane_range[:-1]:
            for t in self.time_range:
                for state in [0] + self.airport_range + [self.n+1]:
                    if P_res[t,state,plane]:
                        plane_line[plane].append((t,state))
                        positions.append((t,state,plane) )

        positions_pass = []
        for r in self.passenger_range:
            positions_pass.append([])
            for t in self.time_range:
                for state in [0] + self.airport_range + [self.n+1]:
                    for plane in self.plane_range:
                        if L_res[r,t,state,plane]:
                            positions_pass[r].append((t,state, plane) )
                            
        positions = np.array(sorted(positions, key = lambda x: x[0]))
        ax[0].scatter(positions[:,0], positions[:,1]+positions[:,2]/5, c=positions[:,2],
                    vmin=0, vmax=len(self.plane_range)-1, alpha=0.5)
        ax[0].set_title("plane positions")
        for r in self.passenger_range:
            positions1 = np.array(positions_pass[r])
            ax[r+1].scatter(positions1[:,0], positions1[:,1]+positions1[:,2]/5, c=positions1[:,2],
                            vmin=0, vmax=len(self.plane_range)-1, alpha=0.5)
            ax[r+1].hlines(y=self.passenger_destinations[r], xmin=0, xmax=np.max(positions1[:,0]), color="black")
            for i, lines in enumerate(plane_line):
                lines = np.array(sorted(lines, key = lambda x: x[0]))
                ax[r+1].plot(lines[:,0], lines[:,1]+i/5, c=["red", "green"][i], alpha=0.5)
            ax[r+1].set_title(f"passenger {r}")


    def solve(self, additional_conditions=None):
        P,L = self.generate_binary_variables()
        self.set_conditions(P, L)
        if additional_conditions is not None:
            additional_conditions(self, P, L)

        model = self.constrain_function(P,L).compile()

        qubo_matrix, variables =self.model_to_matrix(model)

        from gurobi_optimods.qubo import solve_qubo
        result = solve_qubo(qubo_matrix)

        print("Solution to the QUBO problem:", result)
        print("\noptimal")
        P_res_gurobi, L_res_gurobi = self.to_matrix_result( zip(variables, np.array(result.solution)),
                                                           additional_conditions=additional_conditions)
        return P_res_gurobi, L_res_gurobi

planner = QuantumPlanner(number_of_planes=2, number_of_airport=2, number_of_passengers=2, 
                         number_of_time_periods=5,
                         distances= np.array([[0,4,4],[4,0,4],[4,4,0]]),
                         passenger_destinations= np.array([1,2]),
                         passenger_start=np.array([2,1]),
                         airplane_start=np.array([2,1]))

def additional_conditions(planner, P, L):
    pass
    #PRZYKŁAD JAK MOŻNA STOSOWAĆ
    #for t in planner.time_range:
    #    for i in planner.airport_range:
    #        P[t, i, planner.plane_range[-1]] = 1
    #    for i in [0, planner.n+1]: # the plane is never in the air or taking-off
    #        P[t, i, planner.plane_range[-1]] = 0

P_res_gurobi, L_res_gurobi = planner.solve(additional_conditions=additional_conditions)
print(planner.constrain_function(P_res_gurobi, L_res_gurobi, verbose=True))
planner.show_result(P_res_gurobi, L_res_gurobi, figsize=(9,3))
plt.show()