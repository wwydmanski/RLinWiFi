# RLinWiFi
Code for "Contention Window Optimization in IEEE 802.11ax Networks with Deep Reinforcement Learning" article published at WCNC 2021. Preprint available at Arxiv - https://arxiv.org/pdf/2003.01492

## Prerequisites
In order to run this code you need python 3.6 (tensorflow dependency) with installed dependencies:
```
conda env create -f environment.yaml
```
After creating the conda env, **you need a working [ns3-gym](https://github.com/tkn-tub/ns3-gym) environment.** Ns3Gym python package is a part of a larger framework, so installing it on its own is, unfortunately, not enough.

You also require a free ![CometML](https://www.comet.ml/signup) account for viewing the graphs. After creating it update the `comet_token.json` file with your credentials.

## Installation
Clone the repo so that `linear-mesh` directory lands directly in ns3's `scratch`. 

## Execution
All basic configuration of the agents can be done within the file `linear-mesh/agent_training.py` (DDPG) and `linear-mesh/tf_agent_training.py` (DQN).

Benchmark of the static CW values and the original 802.11 backoff algorithm can be set up using `linear-mesh/beb.py` file:

```
usage: beb_tests.py [-h] [--scenario SCENARIOS [SCENARIOS ...]] [--beb]
                    N [N ...]

Run BEB tests

positional arguments:
  N                     number of stations for the scenario (min: 5)

optional arguments:
  -h, --help            show this help message and exit
  --scenario SCENARIOS [SCENARIOS ...]
                        scenarios to run (available: [basic, convergence])
  --beb                 run 802.11 default instead of look-up table
```

Example:
```bash
python agent_training.py                                          # DDPG agent
python tf_agent_training.py                                       # DQN agent
python beb_tests.py --beb 5 10 15 --scenario basic convergence    # Original 802.11 backoff
```

Expected output:
```
Steps per episode: 6000
Waiting for simulation script to connect on port: tcp://localhost:46417
Please start proper ns-3 simulation script using ./waf --run "..."
Waf: Entering directory `/mnt/d/Programy/ns-allinone-3.29/ns-3.29/build'
Waf: Leaving directory `/mnt/d/Programy/ns-allinone-3.29/ns-3.29/build'
Build commands will be stored in build/compile_commands.json
'build' finished successfully (29.428s)
Ns3Env parameters:
--nWifi: 6
--simulationTime: 60
--openGymPort: 46417
--envStepTime: 0.01
--seed: -1
--agentType: continuous
--scenario: convergence
--dryRun: 0
Simulation started
Simulation process id: 20062 (parent (waf shell) id: 20045)
Waiting for Python process to connect on port: tcp://localhost:46417
Please start proper Python Gym Agent
Observation space shape: (1, 300)
Action space shape: (1, 1)
CuDNN version: 7102
cpu
COMET INFO: Experiment is live on comet.ml https://www.comet.ml/XYZ/rl-in-wifi/3ea81647b4a9413182384b646bcef47f

0
  3%|▎         | 182/6300 [00:16<09:22, 10.88it/s, curr_speed=0.00 Mbps, mb_sent=0.00 Mb]
```

## Reading results
Currently, results can only be saved in a [CometML](https://www.comet.ml) workspace. 

Example results for an experiment:
![](https://i.imgur.com/g8hiAz9.png)

## Referencing
You can cite this code as 

```
@INPROCEEDINGS{9417575,
  author={Wydmański, Witold and Szott, Szymon},
  booktitle={2021 IEEE Wireless Communications and Networking Conference (WCNC)}, 
  title={Contention Window Optimization in IEEE 802.11ax Networks with Deep Reinforcement Learning}, 
  year={2021},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/WCNC49053.2021.9417575}}

```
