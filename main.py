import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


#loading data
import loading

our_data = loading.Data("data_simple")


#Now let's check the original schedule

import Scheduler

sched = Scheduler.Scheduler(our_data)
sched._create_transport_orders()
#sched.orders.sort_values("start")

#Let us show you the flight disruptions table
#our_data.flight_disruptions

#We now choose (for now, only one) disruption: first disruption from this table above
#pd.DataFrame(our_data.flight_disruptions.loc[0,:]).transpose()

our_disruptions = [pd.DataFrame(our_data.flight_disruptions.loc[0,:]).transpose()]

sched.apply_flight_disruption(our_disruptions)

leg_id = our_disruptions[0].loc[:,"leg_id"][0] #leg id of disruption
#leg_id

delta = our_disruptions[0].loc[:,"delta"][0] #delta (length of the interval of time) of disruption
#delta

#download list of aircrafts which are affected by disruption
flight_affected = our_data.flights[our_data.flights["leg_id"] == leg_id]
#flight_affected

flights_after_disruption = our_data.flights[(our_data.flights["start"] >= flight_affected.loc[flight_affected.index[0],"start"])]
flights_after_disruption = flights_after_disruption[flights_after_disruption["aircraft"] != "TranspCom#1"]
flights_after_disruption = flights_after_disruption[flights_after_disruption["aircraft"] != "TranspCom#2"]
flights_after_disruption = flights_after_disruption[flights_after_disruption["aircraft"] != "TranspCom#3"]
flights_after_disruption = flights_after_disruption[flights_after_disruption["aircraft"] != "TranspCom#4"]
flights_after_disruption = flights_after_disruption.sort_values("start")
#flights_after_disruption



import datetime
def read_time2(str):
    date, time = str.split(' ')
    year, month, day = date.split('-')
    hour, minute, seconds = time.split(':')
    return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(seconds))

#we assume: 1 interval of time = 30 minutes
def read_moment_time(date_time):
    return 2*date_time.hour + int(date_time.minute/30)


end_time_moment = '2006-01-07 10:10:00'
end_time_moment = read_time2(end_time_moment)
end_time_moment = read_moment_time(end_time_moment)
#end_time_moment



start_time_moment = str(flight_affected.loc[flight_affected.index[0],"start"])
#start_time_moment


start_time_moment = read_time2(start_time_moment)
start_time_moment = read_moment_time(start_time_moment)
#start_time_moment


#We consider moments t in {80, 81, ..., 287} -> number of them: 287-80+1 = 208
number_of_time_periods = end_time_moment-start_time_moment+1
#number_of_time_periods


number_of_passengers = len(flights_after_disruption)-10
#number_of_passengers



aircrafts_after_disruption = list(flights_after_disruption["aircraft"].value_counts().index)[:1]
#aircrafts_after_disruption



number_of_planes = len(aircrafts_after_disruption)
#number_of_planes



airports_from_after_disruption = list(flights_after_disruption["from"].value_counts().index)
airports_to_after_disruption = list((flights_after_disruption["to"].value_counts().index))

airports_after_disruption = list(set(airports_from_after_disruption).union(set(airports_to_after_disruption))-set(['CDG','NCE']))
#airports_after_disruption


number_of_airport = len(airports_after_disruption)
#number_of_airport




#here we have a function which downloads (from data) the distance between two given airports (distance is in minutes)
def d(i = None, j = None, airport_i = None, airport_j = None):
    #i,j = indexes of airports, i.e. i,j are values from {1, ..., n}
    airport_i = our_data.airports.loc[i-1,"code"] if airport_i == None else airport_i
    airport_j = our_data.airports.loc[j-1,"code"] if airport_j == None else airport_j
    route = our_data.routes[(our_data.routes["from"] == airport_i) & (our_data.routes["to"] == airport_j)]
    distance = route["distance"]
    return distance.values[0]


distances = []
for airport_i in airports_after_disruption:
    cur1 = []
    for airport_j in airports_after_disruption:
        if airport_i == airport_j:
            cur1.append(0)
        else:
            cur1.append(int(d(airport_i=airport_i, airport_j=airport_j)/30))
    distances.append(cur1)
distances = np.array(distances)
#distances[0,1] = 3
#distances[1,0] = 3
#distances



#airports_after_disruption


airports_after_disruption_dict = dict()
for i in range(len(airports_after_disruption)):
    airports_after_disruption_dict[airports_after_disruption[i]] = i + 1 
#airports_after_disruption_dict



passenger_destinations = np.array([airports_after_disruption_dict[i] for i in flights_after_disruption.loc[list(flights_after_disruption.index[:2]),'to']])[:2]
#passenger_destinations



passenger_start = np.array([airports_after_disruption_dict[i] for i in flights_after_disruption.loc[list(flights_after_disruption.index[:2]),'from']])[:2]
#passenger_start


airplane_start = []

for airplane in aircrafts_after_disruption:
    temp1 = flights_after_disruption[flights_after_disruption["aircraft"] == airplane]
    _from = temp1.loc[temp1.index[0],"from"]
    airplane_start.append(airports_after_disruption_dict[_from])
#airplane_start



import quantum_planner

planner_simple = quantum_planner.QuantumPlanner(number_of_planes=number_of_planes, number_of_airport=number_of_airport, number_of_passengers=number_of_passengers,\
                                                number_of_time_periods=number_of_time_periods, distances=distances,passenger_destinations=passenger_destinations,\
                                                    passenger_start=passenger_start,airplane_start=airplane_start)



P,L = planner_simple.generate_binary_variables()
planner_simple.set_conditions(P, L)

model = planner_simple.constrain_function(P,L).compile()

qubo_matrix, variables = planner_simple.model_to_matrix(model)

from gurobi_optimods.qubo import solve_qubo
result = solve_qubo(qubo_matrix)

print("Solution to the QUBO problem:", result)
print("\noptimal")
P_res_gurobi, L_res_gurobi = planner_simple.to_matrix_result( zip(variables, np.array(result.solution)))
print(planner_simple.constrain_function(P_res_gurobi, L_res_gurobi, verbose=True))
planner_simple.show_result(P_res_gurobi, L_res_gurobi, figsize=(9,3))
plt.show()