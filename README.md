# RLinWiFi
Code for upcoming research publication.

## Prerequisites
In order to run this code, you must have a working [ns3-gym](https://github.com/tkn-tub/ns3-gym) environment and python 3.6 (and above) with installed dependencies.

## Installation
Clone the repo so that `linear-mesh` directory lands directly in ns3's `scratch`. 

## Execution
All basic configuration can be done within the file `linear-mesh/agent_training.py` (DDPG) and `linear-mesh/tf_agent_training.py` (DQN).
After configuring the scenario, execute python script corresponding to the agent you want to train.

```
python3 linear-mesh/agent_training.py
```
```
python3 linear-mesh/tf_agent_training.py
```

## Reading results
Currently, results can only be saved in a [CometML](https://www.comet.ml) workspace. 

Example results for an experiment:
![](https://i.imgur.com/g8hiAz9.png)
ToDo: add easy CometML token config
