#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:03:46 2023
makes use of steering vector estimation from grid game with no need for reverb like revised_rab2.py
AS of now max step duration of 100 is hardcoded in wrapper, can play with it.
@author: jay
"""

from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
    

from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.drivers import dynamic_step_driver 




from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

from datagen import UpperTriFeatureExtract, DiscreteTargetSpectrum 
import math

num_iterations = 10000#100000 # @param {type:"integer"}



class RAB(py_environment.PyEnvironment):
    
    
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=511, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.int32, minimum=[-60, -60], maximum=[60, 60], name='observation')
        self._state = [0,0]
        
        self._episode_ended = False
        
                
        #observation = (self.theta_B+90,self.theta_M+90)
        
        self.episode_step = 0
        
        #return observation
    
        
    def get_optimal_w(self):
        
        
        self.A = np.array([self.a1, self.a2, self.a3, self.a4], ndmin=2).transpose()
        alpha = 1/np.linalg.multi_dot([self.A.conjugate().transpose(),np.linalg.inv(self.R),self.A])
        self.w = np.dot(alpha*self.R,self.A)
        return self.w
    
    
    def get_B(self):
        
        
        B = np.zeros(np.size(self.spectrum_resolution))
        for theta in self.spectrum_resolution: 
            signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
            #phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[RAB.NUMBER_OF_SENSORS, 1]))
            #steering_vectors = np.exp(1j * phi * (RAB.D_u / RAB.CARRIER_WAVELENGTH))
            temp = np.matmul(self.w.transpose(), self.steering_vectors)
            B[theta] = abs(temp).flatten()
    
        #plt.plot(spectrum_resolution,B)
        
        k = np.where(self.spectrum_resolution == self.lower_bound)
        kk = np.where(self.spectrum_resolution == self.upper_bound)
        c = k[0][0]
        d = kk[0][0]
        self.theta_B = self.spectrum_resolution[c + np.argmax(B[c:d])]
        self._state[0] = self.theta_M
        self._state[1] = self.theta_B
        
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        
        #To start with no priors or ideal A(theta)?
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
        self.episode = 1
        #self.R = R
        
        
        CARRIER_FREQUENCY = 3e9                                       #Frequency of operation
        CARRIER_WAVELENGTH = 3e9/CARRIER_FREQUENCY                    #Lambda value
        NUMBER_OF_SENSORS = 4                                         #Number of sensors
        
        
        NOISE_VARIANCE = 1                                          #Thermal noise variance 
        self.SOI_power = 0.1
        
        #DEVIATION_FACTOR = 0.5                                       #Uncertainty in randomness of nonuniform sensor spacing(add later)
        SNAPSHOTS_PER_REALIZATION = 256                               #Snapshots per iteration, increase this to improve accuracy of BER 
        NUMBER_OF_SOURCES = 1                                         #Number of sources

        self.lower_bound = -60 # For beam patterns to get max 
        self.upper_bound = 60
        inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
        array_length = (NUMBER_OF_SENSORS - 1) * inter_elemental_distance
        uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
        self.spectrum_resolution = np.arange(self.lower_bound, self.upper_bound+1, 1)
        B = np.zeros(np.size(self.spectrum_resolution))
        #Plot the beam pattern 
        D_u = np.tile(uniform_sensor_position,[1, NUMBER_OF_SOURCES])
        #Use correct set of weights here - 
        #wr = np.array([-0.22 - 11j, 0.09 + 0.23j, 0.09 - 0.23j, -0.22 + 11j])
        discrete_spectrum = DiscreteTargetSpectrum()
        data_gen = UpperTriFeatureExtract()
        H = data_gen.hermitian
        self.theta_M = np.random.randint(self.lower_bound, self.upper_bound)
        self._state[0] = self.theta_M
        self._state[1] = 0#(default theta_B-= 0 , your state is 1 x 2 of both thetas. )
        
        signal_dir_rad = data_gen.convert_to_rad(self.theta_M)         
        x_u = data_gen.generate_qpsk_signals(signal_dir_rad=signal_dir_rad, sensor_position=uniform_sensor_position, NUMBER_OF_SENSORS=NUMBER_OF_SENSORS, NOISE_VARIANCE=NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS=SNAPSHOTS_PER_REALIZATION, NUMBER_OF_SOURCES=NUMBER_OF_SOURCES, CARRIER_WAVELENGTH=CARRIER_WAVELENGTH) 
        self.R = x_u.dot(H(x_u))
        
        #w_shifters = ([np.exp(1j*(self.player.p1)*np.pi/180), np.exp(1j*(self.player.p2)*np.pi/180), np.exp(1j*(self.player.p3)*np.pi/180), np.exp(1j*(self.player.p4)*np.pi/180)])/np.sqrt(self.NUMBER_OF_SENSORS)
        
        self.steering_vectors = [self.a1, self.a2, self.a3, self.a4]
        
        self.w = self.get_optimal_w()
        self.SINR = self.get_SINR()
        
                
        #observation = (self.theta_B+90,self.theta_M+90)
        
        self.episode_step = 0
        self._episode_ended = False
        #self._current_time_step = self._reset()
        

        return ts.restart(np.array(self._state, dtype=np.int32))
    
    

    def get_SINR(self):
        
         
        wa = np.dot(self.w.conjugate().transpose(),self.A)
        norm = np.linalg.norm(wa)
        NR = np.square(self.SOI_power) * np.square(norm)
        DR = np.linalg.multi_dot([self.w.conjugate().transpose(),self.R,self.w])        
        SINR = NR/DR
        
        return SINR
        
        
    def get_array_output_power(self):

        power = 1/np.linalg.multi_dot([self.A.conjugate().transpose(),np.linalg.inv(self.R),self.A])
        return power
    
    def move(self):

        if self.c9 == 0:
            
            self.p1 += 0.1*self.c1
            self.p2 += 0.1*self.c2
            self.p3 += 0.1*self.c3
            self.p4 += 0.1*self.c4
            self.p5 += 0.1*self.c5
            self.p6 += 0.1*self.c6
            self.p7 += 0.1*self.c7
            self.p8 += 0.1*self.c8
            
        elif self.c9 == 1: 
            
            self.p1 += -0.1*self.c1
            self.p2 += -0.1*self.c2
            self.p3 += -0.1*self.c3
            self.p4 += -0.1*self.c4
            self.p5 += -0.1*self.c5
            self.p6 += -0.1*self.c6
            self.p7 += -0.1*self.c7
            self.p8 += -0.1*self.c8
            
        else:
            raise ValueError('`action` should be between 0 to 511.')
        # If no value for y, move randomly
        
        variables = [self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8]

        for var in variables:
          
            if var < -1:
                var = -1
            elif var > 1:
                var = 1  
        
        
        self.a1 = self.p1 + 1j*self.p2
        self.a2 = self.p3 + 1j*self.p4
        self.a3 = self.p5 + 1j*self.p6
        self.a4 = self.p7 + 1j*self.p8     
        self.A = self.steering_vectors = [self.a1, self.a2, self.a3, self.a4]
        
        
        
    def _step(self, action):

        if self._episode_ended:
            return self.reset()

    # Make sure episodes don't go on forever.
    
        ans = '{0:09b}'.format(action)
        self.c1,self.c2,self.c3,self.c4,self.c5,self.c6,self.c7,self.c8,self.c9 =[ int(i) for i in [*ans]]
        self.move()
        self.episode += 1
        self.get_B()
        new_SINR = self.get_SINR()
        
        if new_SINR > self.SINR:
            
            self.reward = 5
            
        else:
            
            self.reward = -1
        
        if self._episode_ended or self.episode >= num_iterations:
            #reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), self.reward, discount=1.0)

        
      
    
env = RAB()
time_step = env.reset()
utils.validate_py_environment(env, episodes=18)

tf_env = tf_py_environment.TFPyEnvironment(env)
#Why are we DOING TJHIS STEP? we later create train_env and eval)env


display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()


initial_collect_steps = 10  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 10#100000  # @param {type:"integer"}

batch_size = 64#64 or 5 # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000#100  # @param {type:"integer"} #now code gets stuck after eval_interval

#env_name = 'CartPole-v0'
#env = suite_gym.load(env_name)

env.reset()
#PIL.Image.fromarray(env.render())

#train_py_env = suite_gym.load(env)
#eval_py_env = suite_gym.load(env)

train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(env)


#What should be size of our fc???
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

def compute_avg_return(environment, policy, num_episodes):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0
    i=0#max time step is 100

    while  i < 100:
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
      total_return += episode_return
      i +=1

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]



temp = compute_avg_return(eval_env, random_policy, num_eval_episodes)

table_name = 'uniform_table'
replay_buffer_capacity = 10 


replay_buffer = TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

replay_observer = [replay_buffer.add_batch]

dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
    
iterator = iter(dataset)


train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
]

driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=replay_observer + train_metrics,
    num_steps=1)


'''
try:
  %%time
except:
  pass
'''
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = env.reset()

episode_len = []
reward_list = []

final_time_step, policy_state = driver.run()

for i in range(num_iterations):
    #Line getting stuck always on (i = num_iterations - 3)
    final_time_step, _ = driver.run(final_time_step, policy_state)

    experience, _ = next(iterator)
    train_loss = agent.train(experience=experience)
    step = agent.train_step_counter.numpy()
    if i == 198:
        a = 1
    
    print('Completed epsiode number', i)
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))
        episode_len.append(train_metrics[3].result().numpy())
        print('Average episode length: {}'.format(train_metrics[3].result().numpy()))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        reward_list.append(avg_return)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))


plt.plot(reward_list)
plt.show()
