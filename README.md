# DRL - Project 1
Coursework 1 for the Udacity DRL Nanodegree

## Project details 
This project is completed in the `Banana` environment, where the objective is to maneuver in the space whilst collecting 
yellow bananas and avoiding blue ones. The agent is given a reward of +1 for each yellow banana collected and -1 for 
every blue banana collected.

The state space of the environment is 37 dimensional, consisting of the agents velocity along with ray based perception.
The action space consists of 4 actions: `0 - move forward`, `1 - move backward`, `2 -  turn left`, `3 - turn right`. The 
environment is considered solved once an agent can average a score of +13 over 100 consecutive runs. 

## Getting started
To run this project, all that is required is the `Banana.app` directory provided by Udacity. This 
directory should be placed unzipped alongside the files within this project. Once this is complete, you can look at or 
run (takes a LONG time) the cells in `testing.ipynb` to check the outputs of miscellaneous hyperparameter tests, or 
`train_agent.ipynb` to see the code that trained and saved the final agent. To view
the final report and results, you can open the `Report.ipynb` notebook.