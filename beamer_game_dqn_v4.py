# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:48:23 2023

@author: jkris
"""

import tensorflow as tf
#import keras.backend.tensorflow_backend as backend

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np


import math
import time
import random
from tqdm import tqdm
import os
from PIL import Image
#import cv2


REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "frog1_5k_ep"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20


# Environment settings
EPISODES = 5_000#20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter 

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
    
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        
        model.add(Dense(12, input_shape=(3,)))
        model.add(Activation('relu')) # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/150
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/150
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/150, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.expand_dims(np.array(state, dtype=np.float32), 0)/150)[0]#Size needs to be fized


class Beamer:
    def __init__(self, size):
        self.size = size
        self.p1 = np.random.uniform(-180, 180)
        self.p2 = np.random.uniform(-180, 180)
        self.p3 = np.random.uniform(-180, 180)
        self.p4 = np.random.uniform(-180, 180)

    def __str__(self):
        return f"Blob ({self.p1}, {self.p2},{self.p3}, {self.p4})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        '''
        #Might have to come back to this?
        Do we define theta_B - theta_M ?
        Does that mean theta_B needs to be defined in __init__?
        '''
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(p1 = 0, p2 = 0,p3 = 0, p4 = 0)
        elif choice == 1:
            self.move(p1 = 0, p2 = 0,p3 = 0, p4 = -10)
        elif choice == 2:
            self.move(p1 = 0, p2 = 0,p3 = -10, p4 = 0)
        elif choice == 3:
            self.move(p1 = 0, p2 = 0,p3 = -10, p4 = -10)
        elif choice == 4:
            self.move(p1 = 0, p2 = -10,p3 = 0, p4 = 0)
        elif choice == 5:
            self.move(p1 = 0, p2 = -10,p3 = 0, p4 = -10)
        elif choice == 6:
            self.move(p1 = 0, p2 = -10,p3 = -10, p4 = 0)
        elif choice == 7:
            self.move(p1 = 0, p2 = -10,p3 = -10, p4 = -10) 
        elif choice == 8:
            self.move(p1 = -10, p2 = 0,p3 = 0, p4 = 0)
        elif choice == 9:
            self.move(p1 = -10, p2 = 0,p3 = 0, p4 = -10)
        elif choice == 10:
            self.move(p1 = -10, p2 = 0,p3 = -10, p4 = 0)
        elif choice == 11:
            self.move(p1 = -10, p2 = 0,p3 = -10, p4 = -10) 
        elif choice == 12:
            self.move(p1 = -10, p2 = -10,p3 = 0, p4 = 0)
        elif choice == 13:
            self.move(p1 = -10, p2 = -10,p3 = 0, p4 = -10)
        elif choice == 14:
            self.move(p1 = -10, p2 = -10,p3 = -10, p4 = 0)
        elif choice == 15:
            self.move(p1 = -10, p2 = -10,p3 = -10, p4 = -10)
        elif choice == 16:
            self.move(p1 = 0, p2 = 0,p3 = 0, p4 = 10)
        elif choice == 17:
            self.move(p1 = 0, p2 = 0,p3 = 10, p4 = 0)
        elif choice == 18:
            self.move(p1 = 0, p2 = 0,p3 = 10, p4 = 10)
        elif choice == 19:
            self.move(p1 = 0, p2 = 10,p3 = 0, p4 = 0)
        elif choice == 20:
            self.move(p1 = 0, p2 = 10,p3 = 0, p4 = 10)
        elif choice == 21:
            self.move(p1 = 0, p2 = 10,p3 = 10, p4 = 0)
        elif choice == 22:
            self.move(p1 = 0, p2 = 10,p3 = 10, p4 = 10) 
        elif choice == 23:
            self.move(p1 = 10, p2 = 0,p3 = 0, p4 = 0)
        elif choice == 24:
            self.move(p1 = 10, p2 = 0,p3 = 0, p4 = 10)
        elif choice == 25:
            self.move(p1 = 10, p2 = 0,p3 = 10, p4 = 0)
        elif choice == 26:
            self.move(p1 = 10, p2 = 0,p3 = 10, p4 = 10) 
        elif choice == 27:
            self.move(p1 = 10, p2 = 10,p3 = 0, p4 = 0)
        elif choice == 28:
            self.move(p1 = 10, p2 = 10,p3 = 0, p4 = 10)
        elif choice == 29:
            self.move(p1 = 10, p2 = 10,p3 = 10, p4 = 0)
        elif choice == 30:
            self.move(p1 = 10, p2 = 10,p3 = 10, p4 = 10)

    def move(self, p1=False, p2=False, p3=False, p4=False):

        # If no value for x, move randomly
        if not p1:
            self.p1 += np.random.randint(-10, 10)
        else:
            self.p1 += p1

        # If no value for y, move randomly
        if not p2:
            self.p2 += np.random.randint(-10, 10)
        else:
            self.p2 += p2
            
        if not p3:
            self.p3 += np.random.randint(-10, 10)
        else:
            self.p3 += p3

        # If no value for y, move randomly
        if not p4:
            self.p4 += np.random.randint(-10, 10)
        else:
            self.p4 += p4

        # If we are out of bounds, fix!
        if self.p1 < -180:
            self.p1 = -180
        elif self.p1 > 180:
            self.p1 = 180

        if self.p2 < -180:
            self.p2 = 180
        elif self.p2 > 180:
            self.p2 = 180

        if self.p3 < -180:
            self.p3 = -180
        elif self.p3 > 180:
            self.p3 = 180

        if self.p4 < -180:
            self.p4 = -180
        elif self.p4 > 180:
            self.p4 = 180

class BeamEnv:
    CARRIER_FREQUENCY = 3e9                                       #Frequency of operation
    CARRIER_WAVELENGTH = 3e9/CARRIER_FREQUENCY                    #Lambda value
    NUMBER_OF_SENSORS = 4                                         #Number of sensors
    NOISE_VARIANCE = 1                                          #Thermal noise variance 
    #DEVIATION_FACTOR = 0.5                                       #Uncertainty in randomness of nonuniform sensor spacing(add later)
    SNAPSHOTS_PER_REALIZATION = 256                               #Snapshots per iteration, increase this to improve accuracy of BER 
    NUMBER_OF_SOURCES = 1                                         #Number of sources
    SPECTRUM_WIDTH = 60
    SIZE = 3#Not sure what this is
    OBSERVATION_SPACE_VALUES = (SIZE, 1)
    ACTION_SPACE_SIZE = 31

    MOVE_PENALTY = 1
    ENEMY_PENALTY = 100
    FOOD_REWARD = 500
    SECTOR_REWARD = 250

    lower_bound = -60
    upper_bound = 60

    HPBW = 30

    number_of_sector = abs(lower_bound) + abs(upper_bound) / HPBW

    sector_1 = np.linspace(-60,-30,31)
    sector_2 = np.linspace(-29,0,30)
    sector_3 = np.linspace(1,30,30)
    sector_4 = np.linspace(31,60,30)
      


    lower_bound_2 = -180 # For beam patterns to get max 
    upper_bound_2 = 180


    inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
    array_length = (NUMBER_OF_SENSORS - 1) * inter_elemental_distance
    uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
    uniform_sensor_position = uniform_sensor_position.reshape(NUMBER_OF_SENSORS, 1) #uniform positions for ULA
    spectrum_resolution = np.arange(lower_bound_2, upper_bound_2+1, 1)
    ber_profile = np.arange(0,1.1,0.3)
    weight_profile = np.arange(-1, 1, 0.01)
    angles = np.arange(-180,181,1)
    B = np.zeros(np.size(spectrum_resolution))
    #Plot the beam pattern 
    D_u = np.tile(uniform_sensor_position,[1, NUMBER_OF_SOURCES])
    #Use correct set of weights here - 
    #wr = np.array([-0.22 - 11j, 0.09 + 0.23j, 0.09 - 0.23j, -0.22 + 11j])

    def reset(self):
        self.player = Beamer(self)
        self.theta_M = np.random.randint(BeamEnv.lower_bound, BeamEnv.upper_bound)
        if self.theta_M > 0:
            if self.theta_M/60 > 0.5:
                self.M_sector = 4
            else:
                self.M_sector = 3
            
        else:
            if abs(self.theta_M)/60 > 0.5:
                self.M_sector = 1
            else:
                self.M_sector = 2
        
        
    
        
        self.theta_J = np.random.randint(self.lower_bound, self.upper_bound) #This is jammer theta
        
        w_shifters = ([np.exp(1j*(self.player.p1)*np.pi/180), np.exp(1j*(self.player.p2)*np.pi/180), np.exp(1j*(self.player.p3)*np.pi/180), np.exp(1j*(self.player.p4)*np.pi/180)])/np.sqrt(self.NUMBER_OF_SENSORS)
        
    
        for theta in self.spectrum_resolution: 
            signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
            phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[self.NUMBER_OF_SENSORS, 1]))
            steering_vectors = np.exp(1j * phi * (self.D_u / self.CARRIER_WAVELENGTH))
            temp = np.matmul(w_shifters, steering_vectors)
            self.B[theta] = abs(temp).flatten()
    
        #plt.plot(spectrum_resolution,B)
        
        k = np.where(self.spectrum_resolution == self.lower_bound)
        kk = np.where(self.spectrum_resolution == self.upper_bound)
        c = k[0][0]
        d = kk[0][0]
        self.theta_B = self.spectrum_resolution[c + np.argmax(self.B[c:d])]
        if self.theta_B > 0:
            if self.theta_B/60 > 0.5:
                self.B_sector = 4
            else:
                self.B_sector = 3
            
        else:
            if abs(self.theta_B)/60 > 0.5:
                self.B_sector = 1
            else:
                self.B_sector = 2
                
        observation = (self.theta_B+90,self.theta_M+90,self.theta_J+90)
        
        self.episode_step = 0
        
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        new_observation = (self.theta_B+90,self.theta_M+90,self.theta_J+90)
         
        if self.theta_B == self.theta_J:
             self.reward = -self.ENEMY_PENALTY
        elif self.theta_B == self.theta_M:
             self.reward = self.FOOD_REWARD
        elif self.M_sector == self.B_sector:
             self.reward = self.SECTOR_REWARD
        else:
             self.reward = -self.MOVE_PENALTY
         
        done = False
        if self.reward == self.FOOD_REWARD or self.reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, self.reward, done


    # FOR CNN #
    def get_image(self):
        ip = np.concatenate(([self.theta_B],[self.theta_J,self.theta_J],self.B))
        return ip
            
env = BeamEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
#tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
    
agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    
    