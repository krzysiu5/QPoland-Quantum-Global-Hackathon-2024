import pandas as pd
import os
class Data:
    def __init__(self, directory):
        ##Data about flights
        self.flights = pd.read_csv(os.path.join(directory, "flights.csv"), parse_dates=["date","start", "finish"])
        self.flight_disruptions =  pd.read_csv(os.path.join(directory, "flight_disruptions.csv"))
        self.routes = pd.read_csv(os.path.join(directory, "routes.csv"))
        self.min_ground_time = pd.read_csv(os.path.join(directory, "min_ground_time.csv"))
        
        #below airports columns is a list
        self.maintenance = pd.read_csv(os.path.join(directory, "maintenance.csv"), parse_dates=["min_start", "max_finish"])
        
        #Data about crew
        self.crew = pd.read_csv(os.path.join(directory, "crew.csv"))
        self.crew_unavailabilities = pd.read_csv(os.path.join(directory, "crew_unavailabilities.csv"), parse_dates=["start", "finish"])
        self.crew_pairings = pd.read_csv(os.path.join(directory, "crew_pairings.csv"), parse_dates=["1st_flight_start", "last_flight_finish"])
        self.crew_rostering = pd.read_csv(os.path.join(directory, "crew_rostering.csv"), parse_dates=["start", "finish"])
        self.crew_groups = pd.read_csv(os.path.join(directory, "crew_groups.csv"), parse_dates=["earliest_start_duty"])
        self.min_cabin_crew = pd.read_csv(os.path.join(directory, "min_cabin_crew.csv"))
        self.min_pilots = pd.read_csv(os.path.join(directory, "min_pilots.csv"))
        self.visa_matrix = pd.read_csv(os.path.join(directory, "visa_matrix.csv"))
        
        #Data about airports
        self.airport_closures = pd.read_csv(os.path.join(directory, "airport_closures.csv"), parse_dates=["start", "finish"])
        self.airports = pd.read_csv(os.path.join(directory, "airports.csv"))
        self.aircraft_restrictions = pd.read_csv(os.path.join(directory, "aircraft_restrictions.csv"))
        
        #Data about aircraft
        self.aircraft = pd.read_csv(os.path.join(directory, "aircraft.csv"))
        self.aircraft_unavailabilities = pd.read_csv(os.path.join(directory, "aircraft_unavailabilities.csv"))
        self.cabin_capacities = pd.read_csv(os.path.join(directory, "cabin_capacities.csv"))
        
        #Data about passangers
        self.itineraries = pd.read_csv(os.path.join(directory, "itineraries.csv"))
        self.meal_costs = pd.read_csv(os.path.join(directory, "meal_costs.csv"))
        self.min_con_pax = pd.read_csv(os.path.join(directory, "min_con_pax.csv"))
        
        #Data about slots
        self.slots = pd.read_csv(os.path.join(directory, "slots.csv"), parse_dates=["start", "finish"])
        self.slot_changes = pd.read_csv(os.path.join(directory, "slot_changes.csv"), parse_dates=["start", "finish"])
        