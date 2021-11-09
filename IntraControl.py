from gym import Env,spaces
import numpy as np
# import torch as th
# import numpy as np
from stable_baselines3 import DQN
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.distributions import CategoricalDistribution
# from collections import OrderedDict
# import threading
from stable_baselines3.common.logger import configure

class Sim:
    def __init__(self) -> None:
        pass

    def get_Number_Of_Actions(self):
        return 25
    
    def get_Number_Of_Features(self):
        return 4

    def start(self):

        return obs

    def Execute_Action(self,action):
        ## comm with sim
        ## apply the action(obtain the rew)

        ## ask for the current state
        ## obs [q_i]

        return obs,rew

class IntersectionEnv(Env):
    def __init__(self,sim) -> None:
        super().__init__()
        self.sim=sim

        self.action_space=spaces.Discrete(self.sim.getNumberOfActions())
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(self.sim.get_Number_Of_Features(),))

    def reset(self):
        obs=self.sim.start()
        return obs 

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self,action):

        state,rew=self.sim.Excute_Action(action)
        return state,rew



## define DQN model

env=IntersectionEnv(Sim()) ## Sim need to be implemented 
path="log/"
new_logger=configure(path, ["stdout", "tensorboard"])
model=DQN("MlpPolicy",env,
          learning_rate=5e-3,
          buffer_size=15000,
          learning_starts=200,
          batch_size=32,
          gamma=0.8,
          train_freq=4,
          target_update_interval=300,
          verbose=1,
          tensorboard_log=path,)

model.train(2e4)
model.save("models/m1")
del model

## load model
trainedModel=DQN.load("models/m1")
cycles=100

sim=Sim()## Sim need to be implemented 
obs=sim.start()
for i in range(cycles):
  act,st=trainedModel.predict(obs,deterministic=True)
  obs,rew=sim.Execute_Action(act)
  ## rendering all in simulation side




