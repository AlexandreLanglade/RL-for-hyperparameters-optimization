import tensorflow as tf
import gym
import numpy as np
from tensorflow import keras
from gym import error, spaces, utils
from gym.utils import seeding
import time
import random

class RlfhoEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  
  rewards_history = []
  accuracy_history = []


  def __init__(self, test):
    self.test = test
    if (self.test == False):
      (self.trim, self.trlm), (self.teim, self.telm) = keras.datasets.mnist.load_data()
      (self.tric10, self.trlc10), (self.teic10, self.telc10) = keras.datasets.cifar10.load_data()
      self.random_data = random.randint(1,2) 
    else:
      (self.tric100, self.trlc100), (self.teic100, self.telc100) = keras.datasets.cifar100.load_data()

    self.action_space = spaces.Box(low   = np.array([0.0, 0.0, 0.0, 0.0]),
                                   high  = np.array([7.0, 1.0, 1.0, 1.0]),
                                   dtype = np.float32)
    self.observation_space = spaces.Box(low   = np.array([0.0, 0.0]),
                                        high  = np.array([1.0, 3600.0]),
                                        dtype = np.float32)
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 10
    t1 = time.time()
    self.accuracy = self.run()
    self.accuracy_history.append(self.accuracy)
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
      r = action[2]
      self.opti = tf.keras.optimizers.RMSprop(lr=learning_rate,rho=r)
    elif action[0] < 3:
      learning_rate = action[1]
      b1 = action[2]
      b2 = action[3]
      self.opti = tf.keras.optimizers.Adam(lr=learning_rate,beta_1=b1,beta_2=b2)
    elif action[0] < 4:
      learning_rate = action[1]
      r = action[2]
      self.opti = tf.keras.optimizers.Adadelta(lr=learning_rate, rho=r)
    elif action[0] < 5:
      learning_rate = action[1]
      self.opti = tf.keras.optimizers.Adagrad(lr=learning_rate)
    elif action[0] < 6:
      learning_rate = action[1]
      b1 = action[2]
      b2 = action[3]
      self.opti = tf.keras.optimizers.Adamax(lr=learning_rate, beta_1=b1, beta_2=b2)
    else:
      learning_rate = action[1]
      b1 = action[2]
      b2 = action[3]
      self.opti = tf.keras.optimizers.Nadam(lr=learning_rate, beta_1=b1, beta_2=b2)
    
    self.duree_max -= 1

    accuracy_temp = self.accuracy
    speed_temp = self.speed
    t1 = time.time()
    self.accuracy = self.run()
    self.accuracy_history.append(self.accuracy)
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
    self.rewards_history.append(reward)
    return obs, reward, done, {}


  def reset(self):  
    if (self.test == False):
      self.random_data = random.randint(1,2)
    self.opti = tf.keras.optimizers.SGD()
    self.duree_max = 10
    t1 = time.time()
    self.accuracy = self.run()
    self.accuracy_history.append(self.accuracy)
    self.speed = time.time()-t1
    return np.array([self.accuracy, self.speed])

  def run(self):
    if (self.test == False):
      if(self.random_data == 1):
        print("MNIST")
        self.train_images = self.trim
        self.train_labels = self.trlm
        self.test_images = self.teim
        self.test_labels = self.telm
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0
        self.model = keras.Sequential([
          keras.layers.Flatten(input_shape=(28, 28)),
          keras.layers.Dense(100, activation='relu'),
          keras.layers.Dense(10, activation='softmax'),
      ])
      else:
        print("CIFAR10")
        self.train_images = self.tric10
        self.train_labels = self.trlc10
        self.test_images = self.teic10
        self.test_labels = self.telc10
        self.train_images = self.train_images.astype('float32')
        self.test_images = self.test_images.astype('float32')
        self.train_images /= 255
        self.test_images /= 255
        self.train_labels = self.train_labels.flatten()
        self.test_labels = self.test_labels.flatten()
        input_shape = (32,32,3)
        self.model = keras.Sequential([
          keras.layers.Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=input_shape,
                              padding="same"),  
          keras.layers.Conv2D(32, kernel_size=(3, 3),
                              activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2)),
          keras.layers.Dropout(0.4),
          
          keras.layers.Conv2D(64, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=input_shape,
                              padding="same"),  
          keras.layers.Conv2D(64, kernel_size=(3, 3),
                              activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2)),
          keras.layers.Dropout(0.4),
          
          keras.layers.Conv2D(64, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=input_shape,
                              padding="same"),  
          keras.layers.Conv2D(64, kernel_size=(3, 3),
                              activation='relu'),
          keras.layers.MaxPooling2D(pool_size=(2, 2)),
          keras.layers.Dropout(0.4),
          
          keras.layers.Flatten(),
          
          keras.layers.Dense(512, activation='relu'),
          keras.layers.Dropout(0.5),
          
          keras.layers.Dense(10, activation='softmax')
        ])
    else:
      print("CIFAR100")
      self.train_images = self.tric100
      self.train_labels = self.trlc100
      self.test_images = self.teic100
      self.test_labels = self.telc100
      input_shape = (32, 32, 3)
      self.train_images = self.train_images.astype('float32')
      self.test_images = self.test_images.astype('float32')
      self.train_images = self.train_images / 255
      self.test_images = self.test_images / 255
      self.model = keras.Sequential()
      self.model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
      self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
      self.model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
      self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
      self.model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
      self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
      self.model.add(keras.layers.Flatten())
      self.model.add(keras.layers.Dense(256, activation='relu'))
      self.model.add(keras.layers.Dense(128, activation='relu'))
      self.model.add(keras.layers.Dense(100, activation='softmax'))

    self.model.compile(optimizer=self.opti,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    self.model.fit(self.train_images, self.train_labels, validation_split = 0.1, epochs=3)
    test_loss,test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
    return test_acc

  def getRewardHistory(self):
    return self.rewards_history

  def getAccuracyHistory(self):
    return self.accuracy_history
