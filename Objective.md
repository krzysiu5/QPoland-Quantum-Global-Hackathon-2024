## Constraints

#### Obligatory part

 * Passengers need to travel from specific airport to specific airport using a flight that starts not earlier than their intended flight. (`flights.csv`)
 * For every crew member its next flight have to start from the same airport as previous one ended. It might be possible for a crew to travel as passanger, but then they are payed according to deadheading rate.
 * When airport is closed the planes are not allowed to land of takeoff (`airport_closures.csv`)
 * Member of the crew can not work (what it means?) when are unavailable (`crew_unavailabilities.csv`)
 * Aircraft can not be in use (what it means?) when it was disrupted (`aircraft_disruption`)
 * When flight is disrupted its flight duration is longer by delta (`flight_disruptions.csv`)
 * Aircraft have to be in one of specified airports for maintence in the scheduled time. So the flight before (how much before?) it's maintenace needs to land in the correct airport. Additionaly the next flight's take-off needs to be after(how much after?) the maintence is finished (`maintenance.csv`)
 * **Are there any constraints from** `crew_rosterings.csv` **?????**
 

### Additional part

 * On some airport the aircraft is not allowed to land or takeoff (`aircraft_restriction.csv`)
 * Aircraft can not carry too many people and we prefer to not change the seating class of passengers (`cabin_capacities.csv`)
 * Landing and takeoff can only happen during service hours of the airport (`airports.csv`)
 * Landing and takeoff can 
 * Time between landing and takeoff can not be too short (`min_ground_time.csv`)
 * There are some(?) contraints for crew working time??? (`crew_rostering.csv`, `flights.csv`, `crew.csv`, `crew_groups.csv`, `crew_pairings.csv`)
 * The number of non-pilot crew for a flight on a specific aircraft family for a given number of passangers can not be too small. For example (`min_cabin_crew.csv`)
 * For a given flight duration the number of pilots can not be too small (`min_pilots.csv`)
 * The time between landing and takeoff of a given passanger when changing flights can not be too small (`min_con_pax.csv`)
 * **Some constraints about slots and working time of the crew** `slot_changes.csv`, `slot.csv`, `visa.csv`, `crew_groups.csv`
 * 
 * 
 
 
 ## Costs to minimize
  * 
  * 
  * 