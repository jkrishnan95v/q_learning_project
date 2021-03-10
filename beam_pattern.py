#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:14:59 2021

@author: jay
"""

import random
#import pickle
import numpy as np
#import time
import math
#from numpy.linalg import inv
#from datagen import DataGen
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

CARRIER_FREQUENCY = 3e9                                       #Frequency of operation
CARRIER_WAVELENGTH = 3e9/CARRIER_FREQUENCY                    #Lambda value
NUMBER_OF_SENSORS = 4                                         #Number of sensors
NOISE_VARIANCE = 0.5                                          #Thermal noise variance 
#DEVIATION_FACTOR = 0.5                                       #Uncertainty in randomness of nonuniform sensor spacing(add later)
SNAPSHOTS_PER_REALIZATION = 1000000                               #Snapshots per iteration, increase this to improve accuracy of BER 
NUMBER_OF_SOURCES = 1                                         #Number of sources
SPECTRUM_WIDTH = 60
EPISODES = 65535

lower_bound = -90
upper_bound = 90


inter_elemental_distance = 0.5 * CARRIER_WAVELENGTH                    #uniform spacing of half wavelength
array_length = (NUMBER_OF_SENSORS - 1) * inter_elemental_distance
uniform_sensor_position = np.linspace(-array_length/2, array_length/2, NUMBER_OF_SENSORS)
uniform_sensor_position = uniform_sensor_position.reshape(NUMBER_OF_SENSORS, 1) #uniform positions for ULA
spectrum_resolution = np.arange(-90, 91, 1)
ber_profile = np.arange(0,1.1,0.3)
weight_profile = np.arange(-1, 1, 0.3)

B = np.zeros(361)
#Plot the beam pattern 
D_u = np.tile(uniform_sensor_position,[1, NUMBER_OF_SOURCES])
#Use correct set of weights here - 
wr = np.array([0.5 - 0j, 0.5 + 0j, 0.5 - 0j, 0.5 - 0j])

#wr = np.transpose(wr)
angles = np.arange(-180,181,1)
for theta in angles: 
    signal_dir_rad = theta * math.pi/180                             #DOA to estimate   
    phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1]))
    steering_vectors = np.exp(1j * phi * (D_u / CARRIER_WAVELENGTH))
    temp = np.matmul(wr, steering_vectors)
    B[theta] = abs(temp).flatten()

B2 = 20*np.log10(abs(B))    
plt.plot(angles,B)
