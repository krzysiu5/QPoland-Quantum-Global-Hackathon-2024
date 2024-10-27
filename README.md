## Files

flight_view.py - tkinter app to visualize all flights for the complex data sets

quantum_planner.py - implementation of a basic planner that for given airports and passenger groups finds flights that take
passengers from their starting positions to destinations. You can also specify starting positions of the airplanes.
The planner represents the problem in the form of QUBO and solves it using gurobi library.

main.py - runs a quantum planner on the small section of simple data

Lagranges_Eagles_docs_eng.pdf -  description of the QUBO approach used for solving the planning problem.

Link to the video: https://drive.google.com/drive/folders/18c3r9r1ZLlFpW47WElX-2azrb7pO9raH?usp=sharing

## Requirements

Python
 - numpy
 - pandas
 - pyqubo
 - gurobi
