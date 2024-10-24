import loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_time(str):
    date, time = str.split('T')
    year, month, day = date.split('-')
    hour, minute = time.split(':')
    return datetime.datetime(int(year), int(month), int(day), int(hour), int(minute))

class Flight:
    def __init__(self, full_data : loading.Data, passenger_count):
        self.data = full_data
        self.start = None
        self.start_time = None
        self.end = None
        self.end_time = None
        self.duration = None
        self.passenger_count = passenger_count
        self.rank_requirement = pd.read_csv('crew_for_family.csv', sep=";", index_col=0)
        self.crew = []
    
    def set_time(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = (end_time - start_time).seconds()/60
    
    def predict_end_time(self, start_time):
        if self.start is None or self.end is None:
            raise(Exception("Connection is not set. Prediction uses destination and start airport"))
        self.duration = self.data.routes[(self.data.routes["from"] == self.start) & \
                         (self.data.routes["to"] == self.end)]["distance"].iloc[0]
        return start_time + pd.Timedelta(minutes=self.duration)
    
    def set_connection(self, start, end):
        self.start = start
        self.end = end
    
    def try_assign_aircraft(self, aircraft):
        """assigns aircraft, if it's possible it, and then it will return tuple (true, None).
        If it's not possible it returns false, with data about failure
        """
        self.aircraft = aircraft
        unavailab = self.data.aircraft_unavailabilities
        unavailab = unavailab[unavailab["registration"] == self.aircraft]

        failures = dict()

        for i in range(len(unavailab)):
            start_unavailab = read_time(unavailab.loc[i,"start"]) 
            finish_unavailab = read_time(unavailab.loc[i,"finish"])

            if read_time(self.start_time) <= start_unavailab <= read_time(self.end_time):
                failures["start_time_of_unavailab_within_flight_time"] = [read_time(self.start_time), start_unavailab, read_time(self.end_time)]
            if read_time(self.start_time) <= finish_unavailab <= read_time(self.end_time):
                failures["finish_time_of_unavailab_within_flight_time"] = [read_time(self.start_time), finish_unavailab, read_time(self.end_time)]
        
        restricts = self.data.aircraft_restrictions
        if self.aircraft in restricts[restricts["airport"] == self.end]["disallowed_model"].values:
            failures["aircraft_is_banned_at_target_airport"] = [self.aircraft, self.end]

        our_capacity = data.cabin_capacities[data.cabin_capacities["aircraft"] == self.aircraft]
        our_capacity_sum = our_capacity.loc[0,"cap_first"] + our_capacity.loc[0,"cap_business"] + our_capacity.loc[0,"cap_economic"]

        if our_capacity_sum == -3:
            return tuple(True, None)
        
        if self.passenger_count > our_capacity_sum:
            failures["aircraft_does_not_have_enough_sitting_places"] = [self.aircraft, self.passenger_count, our_capacity_sum]
        
        if len(failures) == 0:
            return tuple(True, None)
        else:
            return tuple(False, failures)
    
    def try_assign_crew(self, crew_list):
        """assigns crew, if it's possible, and then it returns (true, None).
        If it's not possible it returns false, with data about failure
        """

        aircraft_data = self.data.aircraft[self.data.aircraft['registration'] == self.aircraft]
        failures = dict()
        
        ranks_needed = self.rank_requirement.loc[aircraft_data['family'].values[0]].to_dict()

        self.crew = crew_list
        for crew_member in self.crew:
            unavailab = self.data.crew_unavailabilities
            unavailab = unavailab[unavailab["crew_id"] == crew_member]

            for i in range(len(unavailab)):
                start_unavailab = read_time(unavailab.loc[i,"start"]) 
                finish_unavailab = read_time(unavailab.loc[i,"finish"])

                if read_time(self.start_time) <= start_unavailab <= read_time(self.end_time):
                    failures["start_time_of_unavailab_within_flight_time"] = [read_time(self.start_time), start_unavailab, read_time(self.end_time)]
                if read_time(self.start_time) <= finish_unavailab <= read_time(self.end_time):
                    failures["finish_time_of_unavailab_within_flight_time"] = [read_time(self.start_time), finish_unavailab, read_time(self.end_time)]
            
            crew_member_data = self.data.crew[self.data.crew['crew_id'] == crew_member]
            
            if crew_member_data['specs'] != aircraft_data['family']:
                failures['crew_member_has_invalid_qualifications_for_aircraft'] = [crew_member, crew_member_data['specs'], aircraft_data['family']]

            if crew_member in ranks_needed.keys():
                ranks_needed[crew_member] -= 1
                print(ranks_needed[crew_member])
                if min(ranks_needed.values()) < 0:
                    failures['exceed_crew_members_for_given_flight'] = [crew_member, crew_member['specs'], aircraft_data['family']]
            if max(ranks_needed.values()) > 0:
                failures['not_enough_crew_members_for_given_flight'] = [crew_member, crew_member['specs'], aircraft_data['family']]
        
        if len(failures) == 0:
            return tuple(True, None)
        else:
            return tuple(False, failures)


        #TODO : check if these restrictions are enough
    
    def reset_aircraft(self):
        self.aircraft = None
        
    def reset_crew(self):
        self.crew = []

    
class Scheduler:
    def __init__(self, data):
        self.data = data
        self._create_transport_orders()
        self.initial_plan = None
        self.current_time = None # All flights starting before this time are fixed
        self.pairings = [] #
    
    def _create_transport_orders(self):
        """
        Creates data about the number of passengers that have to be transported between specific airports
        starting from specific time.
        """
        #count number of tickets for each seating group
        _tmp = self.data.itineraries.pivot(index=["ticket_id", "leg_id"],columns="cabin_class", values="price").reset_index()
        for col in ["B", "E", "F"]: #in case a given seating group didn't appear
            if col not in _tmp.columns:
                _tmp[col] = np.NaN
        _tmp = _tmp.groupby("leg_id").count() 
        _tmp["total"] = _tmp["B"] + _tmp["E"] + _tmp["F"]
        _tmp.sort_values("total")[["B", "E", "F"]]
        
        self.orders =  pd.merge(self.data.flights, _tmp, on="leg_id")[["start","from", "finish", "to", "total"]]
        
    def apply_flight_disruption(self):
        pass
    
    def find_errors(self):
        pass


class Schedule:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.flight_lists = []
        self.time_slots = []
        self.children_of_flights = [] #this is a list which length is the same as the list of flight_lists
        #and for each flight in flight_lists, value in this list in corresponding position is list of indexes of flights
        #which are the children of specified flight
        self.slots_of_each_flight = []
        self.resources_availible = [] #each element of this list is the list of resources avaible at the t-th time slot of each kind
        self.resources_required = [] #each element of this list is the list of resources required at the t-th time slot to carry job at this slot

        #vector 0-1, i.e x[i,t] = 1 if i-th job is started in the t-th time slot, else 0
        #moreover, we assume that each job cannot occupy more than 1 time slot
        self.x = np.array()
        for i in range(len(self.flight_lists)):
            for t in range(len(self.time_slots)):
                if read_time(self.time_slots[t][0]) == read_time(self.flight_lists[i].start_time) & \
                      read_time(self.time_slots[t][1]) == read_time(self.flight_lists[i].end_time):
                    self.x[i,t] = 1
                else:
                    self.x[i,t] = 0

    def aj_started_only_once(self):
        summarised_col = [sum([self.x[job,slot] for job in self.flight_lists]) for slot in self.time_slots]
        return summarised_col == [1]*len(self.flight_lists)
    
    def aj_started_in_order(self):
        #in this function we assume that aj_started_only_once returned True !!!!!!

        #finding corresponding time_slot for each flight firstly
        for i in range(len(self.flight_lists)):
            self.slots_of_each_flight[i] = np.where(self.x[i] == 1)[0]

        for i in range(len(self.flight_lists)):
            if len(self.children_of_flights[i]) != 0:
                for child in self.children_of_flights[i]:
                     if not abs(self.x[i,self.slots_of_each_flight[i]] * self.x[child,self.slots_of_each_flight[child]]) < 10**(-15):
                         return False
        return True

    def aj_enough_resources(self):
        #in this function we assume that aj_started_only_once returned True and also aj_started_in_order did so !!!!!!!
        failures = [] #for each t-th time slot, value of this list at t-th position is tuple (t-th time_slot, resources_required, resources_avaible)

        for i in range(len(self.flight_lists)):
            for t in range(len(self.resources_availible)):
                if not self.x[i,self.slots_of_each_flight[i]] * self.resources_required[t] <= self.resources_availible[t]:
                    failures[t] = tuple(self.time_slots[t], self.resources_required[t],\
                                                      self.resources_availible[t])
                    
        if len(failures) == 0:
            return True
        else:
            return False, failures        
                

if __name__ == "__main__":
    import datetime
    data = loading.Data("data_complex")
    flight = Flight(data, 10)
    flight.set_connection("AJA", "BCN")
    print(flight.predict_end_time(pd.to_datetime(datetime.datetime(2018, 1, 1))) )
         