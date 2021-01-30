import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from gym import error, spaces, utils
from gym.utils import seeding

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
    self.action_space = spaces.Discrete(4)
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 10
    self.accuracy = self.run()


  def step(self, action):
    if action == 0:
      self.opti = tf.keras.optimizers.SGD()
    elif action == 1:
      self.opti = tf.keras.optimizers.Adam()
    elif action == 2:
      self.opti = tf.keras.optimizers.Adadelta()
    else:
      self.opti = tf.keras.optimizers.Adagrad()
    
    self.duree_max -= 1

    accuracy_temp = self.accuracy
    self.accuracy = self.run()

    if self.accuracy > accuracy_temp:
      reward = 1
    else:
      reward = -1

    if self.duree_max <= 0:
      done = True
    else:
      done = False
    
    return self.opti, reward, done


  def reset(self):
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 10
    self.accuracy = self.run()
    return self.opti

  def run(self):
    self.model.compile(optimizer=self.opti,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    self.model.fit(self.train_images, self.train_labels, validation_split = 0.1, epochs=3)
    test_loss,test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
    return test_acc
