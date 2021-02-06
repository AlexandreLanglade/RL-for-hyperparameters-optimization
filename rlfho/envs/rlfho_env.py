import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from gym import error, spaces, utils
from gym.utils import seeding
import time

class RlfhoEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
  train_images = train_images / 255.0
  test_images = test_images / 255.0
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])


  def __init__(self):
    self.action_space = spaces.Box(low   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                   high  = np.array([7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                   dtype = np.float32)
    self.observation_space = spaces.Box(low   = np.array([0.0, 0.0]),
                                        high  = np.array([100.0, 3600.0]),
                                        dtype = np.float32)
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 70
    t1 = time.time()
    self.accuracy = self.run()
    self.speed = time.time()-t1


  def step(self, action):
    if action[0] < 1:
      lr = action[1]
      mom = action[2]
      if action[3] < 0.5:
        nest = False
      else:
        nest = True
      self.opti = tf.keras.optimizers.SGD(learning_rate=lr, momentum=mom, nesterov=nest)
    elif action[0] < 2:
      lr = action[1]
      mom = action[2]
      r = action[4]
      self.opti = tf.keras.optimizers.RMSprop(learning_rate=lr,rho=r,momentum=mom)
    elif action[0] < 3:
      lr = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Adam(learning_rate=lr,beta_1=b1,beta_2=b2)
    elif action[0] < 4:
      lr = action[1]
      r = action[4]
      self.opti = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=r)
    elif action[0] < 5:
      lr = action[1]
      iav = action[7]
      self.opti = tf.keras.optimizers.Adagrad(learning_rate=lr,initial_accumulator_value=iav)
    elif action[0] < 6:
      lr = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=b1, beta_2=b2)
    else:
      lr = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Nadam(learning_rate=lr, beta_1=b1, beta_2=b2)
    
    self.duree_max -= 1

    accuracy_temp = self.accuracy
    speed_temp = self.speed
    t1 = time.time()
    self.accuracy = self.run()
    self.speed = time.time()-t1

    delta_acc = np.abs(self.accuracy - accuracy_temp)
    delta_time = np.abs(self.speed - speed_temp)
    reward = 0
    if self.accuracy > accuracy_temp:
      reward += 300*delta_acc
    else:
      reward -= 300*delta_acc
    if self.speed < speed_temp:
      reward += 100*delta_time
    else:
      reward -= 100*delta_time

    if self.duree_max <= 0:
      done = True
    else:
      done = False
    
    obs = np.array([self.accuracy, self.speed])

    return obs, reward, done, {}


  def reset(self):
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 70
    t1 = time.time()
    self.accuracy = self.run()
    self.speed = time.time()-t1
    return np.array([self.accuracy, self.speed])

  def run(self):
    self.model.compile(optimizer=self.opti,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    self.model.fit(self.train_images, self.train_labels, validation_split = 0.1, epochs=3)
    test_loss,test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
    return test_acc
