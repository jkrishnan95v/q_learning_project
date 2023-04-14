#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:52:22 2023

@author: jay
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

from datagen import UpperTriFeatureExtract, DiscreteTargetSpectrum 

SOI_power = 20
MODEL_NAME = "protos2"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_0#00  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_00#0  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20


# Environment settings
EPISODES = 5#_00#0#20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = False

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

        
        model.add(Dense(12, input_shape=(2,)))#2 for now later add jamer to make it to 3
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
        current_states = np.array([transition[0] for transition in minibatch])/165
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/165
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
        self.model.fit(np.array(X)/165, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.expand_dims(np.array(state, dtype=np.float32), 0)/165)[0]#Size needs to be fized



class Beamer(object):
    def __init__(self, R):
        #self.size = size
        self.p1 = np.random.uniform(-1, 1) 
        self.p2 = np.random.uniform(-1, 1) 
        self.a1 = self.p1 + 1j*self.p2
        self.p3 = np.random.uniform(-1, 1) 
        self.p4 = np.random.uniform(-1, 1) 
        self.a2 = self.p3 + 1j*self.p4
        self.p5 = np.random.uniform(-1, 1) 
        self.p6 = np.random.uniform(-1, 1) 
        self.a3 = self.p5 + 1j*self.p6
        self.p7 = np.random.uniform(-1, 1) 
        self.p8 = np.random.uniform(-1, 1) 
        self.a4 = self.p7 + 1j*self.p8
        self.R = R
        
    
    def __str__(self):
        return f"Beam initialized with steering vectors ({self.a1}, {self.a2},{self.a3}, {self.a4})"

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
        
        ans = '{0:09b}'.format(choice)
        p1,p2,p3,p4,p5,p6,p7,p8,p9 =[ int(i) for i in [*ans]]
        
        if p9 == 0:
            self.move(p1 = 0.1*p1, p2 = 0.1*p2, p3 = 0.1*p3, p4 = 0.1*p4, p5 = 0.1*p5, p6 = 0.1*p6, p7 = 0.1*p7, p8 = 0.1*p8)
            
        else:
            self.move(p1 = -0.1*p1, p2 = -0.1*p2, p3 = -0.1*p3, p4 = -0.1*p4, p5 = -0.1*p5, p6 = -0.1*p6, p7 = -0.1*p7, p8 = -0.1*p8)
         

    def move(self, p1=False, p2=False, p3=False, p4=False, p5=False, p6=False, p7=False, p8=False):

        # If no value for x, move randomly
        
        variables = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8]
        
        for var in variables:
            
            if var < -1:
                var = -1
            elif var > 1:
                var = 1
        
    def get_optimal_w(self):
        
        
        self.A = np.array([self.a1, self.a2, self.a3, self.a4], ndmin=2).transpose()
        alpha = 1/np.linalg.multi_dot([self.A.conjugate().transpose(),np.linalg.inv(self.R),self.A])
        self.w = np.dot(alpha*self.R,self.A)
        return self.w

    def get_SINR(self):
        
         
        wa = np.dot(self.w.conjugate().transpose(),self.A)
        norm = np.linalg.norm(wa)
        NR = np.square(SOI_power) * np.square(norm)
        DR = np.linalg.multi_dot([self.w.conjugate().transpose(),self.R,self.w])        
        SINR = NR/DR
        
        return SINR
        
        
    def get_array_output_power(self):

        power = 1/np.linalg.multi_dot([self.A.conjugate().transpose(),np.linalg.inv(self.R),self.A])
        return power

class BeamEnv():
    
    
    
    ACTION_SPACE_SIZE = 511
    lower_bound = -60
    upper_bound = 60
    MOVE_PENALTY = 1
    small_reward = 1
    
    def reset(self):
        
        CARRIER_FREQUENCY = 3e9                                       #Frequency of operation
        CARRIER_WAVELENGTH = 3e9/CARRIER_FREQUENCY                    #Lambda value
        NUMBER_OF_SENSORS = 4                                         #Number of sensors
        NOISE_VARIANCE = 1                                          #Thermal noise variance 
        #DEVIATION_FACTOR = 0.5                                       #Uncertainty in randomness of nonuniform sensor spacing(add later)
        SNAPSHOTS_PER_REALIZATION = 256                               #Snapshots per iteration, increase this to improve accuracy of BER 
        NUMBER_OF_SOURCES = 1                                         #Number of sources
        SPECTRUM_WIDTH = 75
        SIZE = 3#Not sure what this is
        OBSERVATION_SPACE_VALUES = (SIZE, 1)
        

        
        

        HPBW = 30


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
        discrete_spectrum = DiscreteTargetSpectrum()
        data_gen = UpperTriFeatureExtract()
        H = data_gen.hermitian
        self.theta_M = np.random.randint(BeamEnv.lower_bound, BeamEnv.upper_bound)
        
        
        signal_dir_rad = data_gen.convert_to_rad(self.theta_M)         
        x_u = data_gen.generate_qpsk_signals(signal_dir_rad=signal_dir_rad, sensor_position=uniform_sensor_position, NUMBER_OF_SENSORS=NUMBER_OF_SENSORS, NOISE_VARIANCE=NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS=SNAPSHOTS_PER_REALIZATION, NUMBER_OF_SOURCES=NUMBER_OF_SOURCES, CARRIER_WAVELENGTH=CARRIER_WAVELENGTH) 
        self.R = x_u.dot(H(x_u))
        
        self.player = Beamer(self.R)

        '''
        Include jammer later stages
        while True:    
            self.theta_J = np.random.randint(self.lower_bound, self.upper_bound) #This is jammer theta
            if self.theta_J != self.theta_M:
                break
        '''    
        
        w_shifters = self.player.get_optimal_w()
        #w_shifters = ([np.exp(1j*(self.player.p1)*np.pi/180), np.exp(1j*(self.player.p2)*np.pi/180), np.exp(1j*(self.player.p3)*np.pi/180), np.exp(1j*(self.player.p4)*np.pi/180)])/np.sqrt(self.NUMBER_OF_SENSORS)
        
    
        for theta in spectrum_resolution: 
            signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
            phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
            steering_vectors = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH))
            temp = np.matmul(w_shifters.transpose(), steering_vectors)
            B[theta] = abs(temp).flatten()
    
        #plt.plot(spectrum_resolution,B)
        
        k = np.where(spectrum_resolution == self.lower_bound)
        kk = np.where(spectrum_resolution == self.upper_bound)
        c = k[0][0]
        d = kk[0][0]
        self.theta_B = spectrum_resolution[c + np.argmax(B[c:d])]
        
        self.SINR = self.player.get_SINR()
                
        observation = (self.theta_B+90,self.theta_M+90)
        
        self.episode_step = 0
        
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        new_observation = (self.theta_B+90,self.theta_M+90)
        w_shifters = self.player.get_optimal_w()
        new_SINR = self.player.get_SINR()

        
        if new_SINR > self.SINR:
            self.reward = self.small_reward
        else:
            self.reward = -self.MOVE_PENALTY
        
        self.SINR = new_SINR
        return new_observation, self.reward


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

        new_state, reward = env.step(action)

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
    
    