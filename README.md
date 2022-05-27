<img width="350" src="./resources/logo.png" align="right" >

# ADCS Simulations
### From TEIDESAT Project and Hyperspace Canarias

## Description
This repository contains the simulations of the Attitude Determination and Control System (ADCS) for the TEIDESAT Project and will consist of simulating the Detumbling, Pointing and Tracking phases.

## Execution
To execute the simulations, you need to have the following packages installed:
* [Numpy] (https://numpy.org/)
* [Matplotlib] (https://matplotlib.org/)
* [Scipy] (https://www.scipy.org/)
* [Pyatmos] (https://pypi.org/project/pyatmos/)

To run the program, you need to execute the following command:
```
python ./main.py
```
This will start running the Detumbling simulation. For the moment, to change
the simulation time, you can change the value of the constant `SECONDS` in 
the file `constants.py`.

## Current state
The simulations are currently being refactored and will be updated soon.
To the moment only the Detumbling phase is implemented and is being tested.

### TODO
* Tests will be needed for every function.
* The documentation can be improved, and needs to be checked for theory errors (variables, units, etc).
* Constants could be imported one by one, according to the needs of each function.
* Some optional arguments for the main file could be added, such as the possibility to change the simulation time.