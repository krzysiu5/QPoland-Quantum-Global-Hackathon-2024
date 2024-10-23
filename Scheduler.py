import loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def odczytaj_czas(str):
    data, czas = str.split('T')
    rok, miesiac, dzien = data.split('-')
    godzina, minuta = czas.split(':')
    return datetime.datetime(int(rok), int(miesiac), int(dzien), int(godzina), int(minuta))

class Flight:
    def __init__(self, full_data : loading.Data, passenger_count):
        self.data = full_data
        self.start = None
        self.start_time = None
        self.end = None
        self.end_time = None
        self.duration = None
        self.passenger_count = passenger_count
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
        unavilib = self.data.aircraft_unavailabilities
        unavilib = unavilib[unavilib["Registration"] == self.aircraft]

        niepowodzenia = dict()

        for i in range(len(unavilib)):
            start_dostep = odczytaj_czas(unavilib.loc[i,"start"]) 
            finish_dostep = odczytaj_czas(unavilib.loc[i,"finish"])

            if odczytaj_czas(self.start_time) <= start_dostep <= odczytaj_czas(self.end_time):
                niepowodzenia["start_time_of_unavilib_within_flight_time"] = [odczytaj_czas(self.start_time), start_dostep, odczytaj_czas(self.end_time)]
            if odczytaj_czas(self.start_time) <= finish_dostep <= odczytaj_czas(self.end_time):
                niepowodzenia["finish_time_of_unavilib_within_flight_time"] = [odczytaj_czas(self.start_time), finish_dostep, odczytaj_czas(self.end_time)]
        
        restricts = self.data.aircraft_restrictions
        if self.aircraft in restricts[restricts["airport"] == self.end]["disallowed_model"].values:
            niepowodzenia["aircraft_is_banned_at_target_airport"] = [self.aircraft, self.end]

        our_capacity = data.cabin_capacities[data.cabin_capacities["aircraft"] == self.aircraft]
        our_capacity_sum = our_capacity.loc[0,"cap_first"] + our_capacity.loc[0,"cap_business"] + our_capacity.loc[0,"cap_economic"]

        if our_capacity_sum == -3:
            return tuple(True, None)
        
        if self.passenger_count > our_capacity_sum:
            niepowodzenia["aircraft_does_not_have_enough_sitting_places"] = [self.aircraft, self.passenger_count, our_capacity_sum]
        
        if len(niepowodzenia) == 0:
            return tuple(True, None)
        else:
            return tuple(False, niepowodzenia)
    
    def try_assign_crew(self, crew_list):
        """assigns crew, if it's possible, and then it returns (true, None).
        If it's not possible it returns false, with data about failure
        """
        self.crew = crew_list
        #TODO : check if crew has qualifications 
    
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


if __name__ == "__main__":
    import datetime
    data = loading.Data("data_complex")
    flight = Flight(data, 10)
    flight.set_connection("AJA", "BCN")
    print(flight.predict_end_time(pd.to_datetime(datetime.datetime(2018, 1, 1))) )
         