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
    self.action_space = spaces.Box(low   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                   high  = np.array([7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                                   dtype = np.float32)
    self.observation_space = spaces.Box(low   = np.array([0.0, 0.0]),
                                        high  = np.array([1.0, 3600.0]),
                                        dtype = np.float32)
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 10
    t1 = time.time()
    self.accuracy = self.run()
    self.speed = time.time()-t1


  def step(self, action):
    if action[0] < 1:
      learning_rate = action[1]
      mom = action[2]
      if action[3] < 0.5:
        nest = False
      else:
        nest = True
      self.opti = tf.keras.optimizers.SGD(lr=learning_rate, momentum=mom, nesterov=nest)
    elif action[0] < 2:
      learning_rate = action[1]
      r = action[4]
      self.opti = tf.keras.optimizers.RMSprop(lr=learning_rate,rho=r)
    elif action[0] < 3:
      learning_rate = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Adam(lr=learning_rate,beta_1=b1,beta_2=b2)
    elif action[0] < 4:
      learning_rate = action[1]
      r = action[4]
      self.opti = tf.keras.optimizers.Adadelta(lr=learning_rate, rho=r)
    elif action[0] < 5:
      learning_rate = action[1]
      self.opti = tf.keras.optimizers.Adagrad(lr=learning_rate)
    elif action[0] < 6:
      learning_rate = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Adamax(lr=learning_rate, beta_1=b1, beta_2=b2)
    else:
      learning_rate = action[1]
      b1 = action[5]
      b2 = action[6]
      self.opti = tf.keras.optimizers.Nadam(lr=learning_rate, beta_1=b1, beta_2=b2)
    
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
    self.duree_max = 10
    t1 = time.time()
    self.accuracy = self.run()
    self.speed = time.time()-t1
    return np.array([self.accuracy, self.speed])

  def run(self):
    self.model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
    ])
    self.model.compile(optimizer=self.opti,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    self.model.fit(self.train_images, self.train_labels, validation_split = 0.1, epochs=3)
    test_loss,test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
    return test_acc
