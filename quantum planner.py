from pyqubo import Binary
import numpy as np
import re
a, b, c, d = Binary("a"), Binary("b"), Binary("c"), Binary("d")
model = (a*b*c + a*b*d).compile()
print(model.to_qubo())

n=3
plane_range = list(range(2))
airport_range = list(range(1,n+1))
passenger_range = list(range(3))
time_range = list(range(5))

distances = np.array([[1,2,1],[2,1,1],[1,1,1]])
distances

passenger_destinations = np.array([0,1,2])

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
L.shape

###!!!! Można przypisać wartości, które moją być ustalone bezpośrednio !!!
for r in passenger_range:
    L[r, time_range[-1], passenger_destinations[r], 0] = 1

only_one_state = np.sum((np.sum(P, axis=1) - 1)**2)

_to_sum = []
for t1 in time_range:
    for t2 in time_range:
        for i in airport_range:
            for j in airport_range:
                if abs(t1-t2)<distances[i-1,j-1]:
                    _to_sum.append(P[t1,i,:]*P[t2,j,:])
cond_P1 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    _to_sum.append(np.array(P[t,1:n+1,:]*P[t+1,0:1,:]))
cond_P2 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    _to_sum.append(np.array(P[t,n+1,:]*P[t+1,1:n+2,:]))
cond_P3 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    _to_sum.append(np.array(P[t,0,:]*P[t+1,n+1,:]))
cond_P4 = np.sum(_to_sum)

conditions_on_P = only_one_state + cond_P1 + cond_P2 + cond_P3 + cond_P4



only_one_place = np.sum((np.sum(np.sum(L, axis=-1), axis=-1) - 1)**2)

P = np.expand_dims(P, 0)

_to_sum = []
for r in passenger_range:
    for i in set([0,n+1]) | set(airport_range) - set([passenger_destinations[r]]):
        _to_sum.append(np.sum(-L[r,:,i,:]*P[0,:,i,:]))
cond1 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    for i in [0] + airport_range:
        _to_sum.append(np.sum(-L[:,t,0,:]*L[:,t+1,i,:]))
cond2 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    for u in plane_range:
        for k in plane_range:
            _to_sum.append(np.sum(-L[:, t, 1:n+1, u] * L[:, t+1, n+1:n+2, k]))
cond3 = np.sum(_to_sum)

_to_sum = []
for r in passenger_range:
    for t in time_range[:-1]:
        for i in set(airport_range) - set([passenger_destinations[r]]):
            _to_sum.append(np.sum(-L[r, t, i, :] * L[r, t+1, n+1, :]))
cond4 = np.sum(_to_sum)

_to_sum = []
for t in time_range[:-1]:
    _to_sum.append(np.sum(L[:, t, n+1, :] * L[:, t+1, 0, :]))
cond5 = np.sum(_to_sum)

_to_sum = []
for r in passenger_range:
    _x = passenger_destinations[r]
    for t in time_range[:-1]:
        for u in plane_range:
            for k in plane_range:
                _to_sum.append(np.sum(L[r, t, _x, u] * L[r, t+1, _x, k]))
cond6 = 2*np.sum(_to_sum)

conditions_on_LP = only_one_place + cond1 + cond2 + \
                   cond3 + cond4 + cond5 + cond6

H = conditions_on_LP + conditions_on_P
model = H.compile()
bqm = model.to_bqm()
import neal
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
#print(best_sample.sample)

L_res = np.zeros([len(passenger_range), 
                  len(time_range),
                  len(airport_range)+2,
                  len(plane_range)])
P_res = np.zeros([len(time_range),
                  len(airport_range)+2,
                  len(plane_range)])
for key, val in  best_sample.sample.items():
    if key[0] == "P":
        time, state, plane = re.match("P_t(\d*),A(\d*),B(\d*)", key).groups()
        time, state, plane = int(time), int(state), int(plane)
        P_res[time, state, plane] = val
    if key[0] == "L":
        passenger, time, state, plane = re.match("L_C(\d*),t(\d*),A(\d*),B(\d*)", key).groups()
        passenger, time, state, plane = int(passenger), int(time), int(state), int(plane)
        L_res[passenger, time, state, plane] = val

for plane in plane_range:
    print("plane ", plane)
    for t in time_range:
        print("  at time", t)
        for state in [0] + airport_range + [n+1]:
            if P_res[t,state,plane]:
                print(f"    is in state {state}")

for r in passenger_range:
    print("passenger",r)
    for t in time_range:
        print("  time", t)
        for state in [0] + airport_range + [n+1]:
            for plane in plane_range:
                if L_res[r,t,state,plane]:
                    print(f"    is in plane {plane} and state {state}")
    L_res[0][0]