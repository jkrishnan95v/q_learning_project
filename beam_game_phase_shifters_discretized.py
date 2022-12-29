# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:14:59 2021
@author: jay
"""
import datagen
import random
import pickle
import numpy as np
import time
import math
#from numpy.linalg import inv
#from datagen import DataGen
import matplotlib.pyplot as plt
from matplotlib import style

import io
import cv2
from PIL import Image

style.use("ggplot")

CARRIER_FREQUENCY = 3e9                                       #Frequency of operation
CARRIER_WAVELENGTH = 3e9/CARRIER_FREQUENCY                    #Lambda value
NUMBER_OF_SENSORS = 4                                         #Number of sensors
NOISE_VARIANCE = 1                                          #Thermal noise variance 
#DEVIATION_FACTOR = 0.5                                       #Uncertainty in randomness of nonuniform sensor spacing(add later)
SNAPSHOTS_PER_REALIZATION = 256                               #Snapshots per iteration, increase this to improve accuracy of BER 
NUMBER_OF_SOURCES = 1                                         #Number of sources
SPECTRUM_WIDTH = 60


EPISODES = 20000
SHOW_EVERY = 1000


epsilon = 0.9
LEARING_RATE = 1
DISCOUNT = 0.95
MOVE_PENALTY = 1
ENEMY_PENALTY = 100
FOOD_REWARD = 100
SECTOR_REWARD = 5

EPS_DECAY = 0.9998  



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




'''
#wr = np.transpose(wr)
angles = np.arange(-180,181,1)
for theta in spectrum_resolution: 
    signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
    phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
    steering_vectors = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH))
    temp = np.matmul(wr, steering_vectors)
    B[theta] = abs(temp).flatten()

#B2 = 20*np.log10(abs(B))    
plt.plot(angles,B)
'''

start_q_table = None
#start_q_table = "qtable-1665426677.pickle" #We can load a previously trained q table



class Beamer:
    def __init__(self):
        self.p1 = np.random.uniform(-180, 180)
        self.p2 = np.random.uniform(-180, 180)
        self.p3 = np.random.uniform(-180, 180)
        self.p4 = np.random.uniform(-180, 180)
                
        
    def __str__(self):
        return f"{self.p1}, {self.p2},{self.p3}, {self.p4}"
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
        
    def action(self, choice):
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
        if  p1:
            self.p1 += p1
        if  p2:
            self.p2 += p2   
            
        if  p3:
            self.p3 += p3
        if  p4:
            self.p4 += p4    
                   
            
        if self.p1 < -180:
            self.p1 = -180
        elif self.p1 > 180:
            self.p1 = 180

        if self.p2 < -180:
            self.p2 = -180
        elif self.p2 > 180:
            self.p2 = -180

        if self.p3 < -180:
            self.p3 = -180
        elif self.p3 > 180:
            self.p3 = 180

        if self.p4 < -180:
            self.p4 = -180
        elif self.p4 > 180:
            self.p4 = -180


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



if start_q_table is None:
    q_table = {} 
    for x1 in range(2*lower_bound, 2*upper_bound):
        for y1 in range(2*lower_bound, 2*upper_bound):
                    q_table[(x1, y1)] = [np.random.uniform(-1, 1) for i in range(31)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

                    
episode_rewards = []
steps = 100

for episode in range(EPISODES):
    beam = Beamer()
    theta_M = np.random.randint(lower_bound, upper_bound)#This is your SOI
    if theta_M > 0:
        if theta_M/60 > 0.5:
            sector_M = 4
        else:
            sector_M = 3
        
    else:
        if abs(theta_M)/60 > 0.5:
            sector_M = 1
        else:
            sector_M = 2
    
    

    while True:
        theta_J = np.random.randint(lower_bound, upper_bound) #This is jammer theta
        if theta_J != theta_M:
            break    
    
    
    w_shifters = ([np.exp(1j*beam.p1*np.pi/180), np.exp(1j*beam.p2*np.pi/180), np.exp(1j*beam.p3*np.pi/180), np.exp(1j*beam.p4*np.pi/180)])/np.sqrt(NUMBER_OF_SENSORS)
    
    
    for theta in spectrum_resolution: 
        signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
        phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
        steering_vectors = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH))
        temp = np.matmul(w_shifters, steering_vectors)
        B[theta] = abs(temp).flatten()

    plt.plot(spectrum_resolution,B)
    
    k = np.where(spectrum_resolution == lower_bound)
    kk = np.where(spectrum_resolution == upper_bound)
    c = k[0][0]
    d = kk[0][0]
    theta_B = spectrum_resolution[c + np.argmax(B[c:d])]# This is your beam theta
    if theta_B > 0:
        if theta_B/60 > 0.5:
            sector_B = 4
        else:
            sector_B = 3
        
    else:
        if abs(theta_B)/60 > 0.5:
            sector_B = 1
        else:
            sector_B = 2
    
    
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    
    else:
        show = False
        
    episode_reward = 0
    
    for i in range(steps):
        obs = (theta_B - theta_M, theta_B - theta_J)
        
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs]) 
        else:
            action = np.random.randint(0, 31)
            
        beam.action(action)
        '''Disabled moving jammer for now
        theta_J += theta_J + np.random.randint(-1,2)
        if theta_J < lower_bound:
            theta_J = lower_bound
        elif theta_J > upper_bound:
            theta_J = upper_bound
        
        theta_M += theta_M + np.random.randint(-1,2)
        if theta_M < lower_bound:
            theta_M = lower_bound
        elif theta_M > upper_bound:
            theta_M = upper_bound
        if theta_M > 0:
            if theta_M/60 > 0.5:
                sector_M = 4
            else:
                sector_M = 3
            
        else:
            if abs(theta_M)/60 > 0.5:
                sector_M = 1
            else:
                sector_M = 2
        '''
        w_shifters = ([np.exp(1j*beam.p1*np.pi/180), np.exp(1j*beam.p2*np.pi/180), np.exp(1j*beam.p3*np.pi/180), np.exp(1j*beam.p4*np.pi/180)])/np.sqrt(NUMBER_OF_SENSORS)
        
        
        for theta in spectrum_resolution: 
            signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
            phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
            steering_vectors = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH))
            temp = np.matmul(w_shifters, steering_vectors)
            B[theta] = abs(temp).flatten()
        
        k = np.where(spectrum_resolution == lower_bound)
        kk = np.where(spectrum_resolution == upper_bound)
        c = k[0][0]
        d = kk[0][0]
        theta_B = spectrum_resolution[c + np.argmax(B[c:d])]
        
        if theta_B > 0:
            if theta_B/60 > 0.5:
                sector_B = 4
            else:
                sector_B = 3
            
        else:
            if abs(theta_B)/60 > 0.5:
                sector_B = 1
            else:
                sector_B = 2
        
        
        if theta_B == theta_J:
            reward = -ENEMY_PENALTY
        elif theta_B == theta_M:
            reward = FOOD_REWARD
        elif sector_M == sector_B:
            reward = SECTOR_REWARD
        else:
            reward = -MOVE_PENALTY
        
        
        w_shifters = ([np.exp(1j*beam.p1*np.pi/180), np.exp(1j*beam.p2*np.pi/180), np.exp(1j*beam.p3*np.pi/180), np.exp(1j*beam.p4*np.pi/180)])/np.sqrt(NUMBER_OF_SENSORS)
        
        
       
        new_obs = (theta_B - theta_M, theta_B - theta_J)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        elif reward == SECTOR_REWARD:
            new_q = SECTOR_REWARD
        else:
            new_q = (1 - LEARING_RATE) * current_q + LEARING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
        
        '''
        if show:
            
            plt.clf
            plt.plot(angles,B)
            plt.axvline(x=theta_M, color = 'blue')
            plt.axvline(x=theta_J, color = 'green')
            plot_img_np = get_img_from_fig(plt)
            
            img = Image.fromarray(plot_img_np, "RGB")
            #img = img.resize((300,300))
            cv2.imshow("The quintessial chase", np.array(img))
            
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        '''
        episode_reward += reward
        if reward == FOOD_REWARD: #or reward == SECTOR_REWARD:
            break
        
        #What if you tell not to break from reward == - enemy_penalty, will it eventually learn to get to reward. 
        
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
    
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)

plt.ylabel(f"reward{SHOW_EVERY} ")
plt.xlabel("epsiode #")
plt.show
plt.savefig('foo.png')

with open (f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
    
