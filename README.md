# RLinWiFi
Code for "Contention Window Optimization in Wi-Fi Networks with Deep Reinforcement Learning" publication

## Pre-requisites
In order to run this code, you must have a working [ns3-gym](https://github.com/tkn-tub/ns3-gym) environment.

## Installation
Clone the repo so that `linear-mesh` directory lands directly in ns3's `scratch`. 

## Execution
All basic configuration can be done within the file `linear-mesh/agent_training.py` (DDPG) and `linear-mesh/tf_agent_training.py` (DQN).
After configuring the scenario, execute python script corresponding to the agent you want to train.

## Reading results
Currently, results can only be saved in a [CometML](https://www.comet.ml) workspace. 

Example results for an experiment:
![](https://i.imgur.com/g8hiAz9.png)
ToDo: add easy CometML token config
