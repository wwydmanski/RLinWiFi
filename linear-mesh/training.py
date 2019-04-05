from ns3gym import ns3env
import subprocess
from agents.ddpg.agent import Agent
from agents.teacher import Teacher

simTime = 10 # seconds
stepTime = 0.1  # seconds

EPISODE_COUNT = 3
port = 5555
seed = 0
simArgs = {"--simTime": simTime,
           "--testArg": 123,
           "--nodeNum": 5,
           "--distance": 500}
debug = False

subprocess.Popen(['bash', '-c', script_exec_command])
env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=0, simSeed=seed, simArgs=simArgs, debug=debug)

env.reset()
ob_space = env.observation_space
ac_space = env.action_space

print("Observation space shape:", ob_space)
print("Action space shape:", ac_space)

assert ob_space is not None

agent = Agent(6, 1)
teacher = Teacher(agent, env, 1)
teacher.train(EPISODE_COUNT, simTime, stepTime)