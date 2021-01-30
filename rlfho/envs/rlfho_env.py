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
    self.action_space = spaces.Discrete(3)
    #self.observation_space = spaces.Box(low=np.array([0.0]),high=np.array([1.0]))
    self.learning_rate = 0.5 #hyperparametre valeur initial
    self.duree_max = 60
    self.accuracy = run()


  def step(self, action):
    # Step the environment by one timestep. Returns observation, reward, done, info.
    self.learning_rate += (action-1)*0.1
    self.duree_max -= 1

    accuracy_temp = self.accuracy
    self.accuracy = run()

    #calcul reward
    if self.accuracy > accuracy_temp:
      reward = 1
    else:
      reward = -1

    if self.duree_max <= 0:
      done = True
    else:
      done = False

    return self.learning_rate, reward, done, self.info


  def reset(self):
    # Reset the environment's state. Returns observation.
    self.learning_rate = 0.5 #hyperparametre valeur initial
    self.duree_max = 60
    self.accuracy = run()
    return self.learning_rate

  def run(self):
    sgd = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
    model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_images, train_labels, validation_split = 0.1, epochs=5)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return test_acc