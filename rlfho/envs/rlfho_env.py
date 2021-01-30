import gym
from gym import error, spaces, utils
from gym.utils import seeding

class RlfhoEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):

    pass
  def step(self, action):
    # Step the environment by one timestep. Returns observation, reward, done, info.
    pass
  def reset(self):
    # Reset the environment's state. Returns observation.
    pass
  def render(self, mode='human'):
    # Render one frame of the environment. The default mode will do something human friendly, such as pop up a window.
    pass
  def close(self):
    pass