    #!/usr/bin/python3

import random
import numpy as np
from numpy import linalg as lalg
import math
from sklearn.preprocessing import MultiLabelBinarizer
#import doatools.estimation as estimation 
from sklearn.metrics import mean_absolute_error



class DataGen:
    """
    Class for generating incoming signal data for sensor array processing

    """
    def __init__(self):
        """ __init__
        Arguments:
            None

        Returns:
            None

        """
        pass
    def generate_random_float(self,lower_bound, upper_bound):
        
        number = (upper_bound- lower_bound) * np.random.random() + lower_bound
        return number
    
    def generate_random_number(self,lower_bound, upper_bound):
        
        number = np.random.randint(lower_bound, upper_bound+1)
        return number

    def generate_random_directions(self, lower_bound, upper_bound, size, scale=None):
        """ Returns randomly generated signals between a given interval, 
            usually the visible spectrum of the sensor array.
        
        Arguments:
            lower_bound (int) : defines the lower bound for a random signals
            upper_bound (int) : defines the upper bound for a random signals
            size (int)        : defines the number of signals
            
        Returns:
            Randomly generated signals between an azimuthal interval
        """
        visible_spectrum = (lower_bound + upper_bound)/2
        if scale is None:
             scale = (upper_bound-lower_bound)/2
        results = []
        while len(results) < size:
            signal_dirs = np.random.normal(loc=visible_spectrum, scale=scale, size=size-len(results))
            signal_dirs = np.round(signal_dirs,1)
            results += [sample for sample in signal_dirs if lower_bound <= sample <= upper_bound]  
        return np.array(results )   

    def hermitian(self, A, **kwargs): ##define hermitian
        """ hermitian
        Arguments:
            A   (complex)  : Complex valued matrix

        Returns:
            Conjugate transpose of the input matrix
        """
        return np.transpose(A, **kwargs).conj()
    
    def convert_to_rad(self, value_degree):
        """ convert_to_rad
        Arguments:
            value_degree  (float)  : angles in degrees

        Returns:
            angles in radians
        """
        return value_degree * math.pi/180
    
    def generate_qpsk_signals(self, signal_dir_rad, sensor_position, NUMBER_OF_SENSORS, NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES, CARRIER_WAVELENGTH):
            phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1])) 
            D_u = np.tile(sensor_position, [1, NUMBER_OF_SOURCES])
            steervec_u = np.exp(1j * phi * (D_u/CARRIER_WAVELENGTH))                            #uniform steering vectors
            symbols = np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) + 1j*np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) #QPSK symbols
            x_u = np.zeros(([NUMBER_OF_SENSORS, NUMBER_OF_SNAPSHOTS]), dtype=complex)
            for i in range(NUMBER_OF_SNAPSHOTS):
                x_u[:,i] = np.sum(np.tile(symbols[i,:],[NUMBER_OF_SENSORS, 1])*steervec_u,1)  #uniformly sampled data
                noise = NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1]) + 1j * NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1])  
                x_u = x_u + noise
            return x_u
        
    def generate_correlated_qpsk_signals(self, signal_dir_rad, sensor_position, NUMBER_OF_SENSORS, NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES, CARRIER_WAVELENGTH, number_of_correlatedsources, correlated_aoa):
            phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1])) 
            pho = np.zeros(number_of_correlatedsources)
            
            pho_aoa = 2*math.pi*np.sin(np.tile(correlated_aoa,[NUMBER_OF_SENSORS, 1])) 
            deltaphi = np.zeros( number_of_correlatedsources)
            
            distorted_symbols = np.zeros((NUMBER_OF_SNAPSHOTS, number_of_correlatedsources))
            symbols = np.zeros((NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES))
            
            total_signals = number_of_correlatedsources + NUMBER_OF_SOURCES
            D_u = np.tile(sensor_position, [1, total_signals])
            
            distorted_symbols = np.zeros((NUMBER_OF_SNAPSHOTS, number_of_correlatedsources))
            symbols = np.zeros((NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES))
            
            angles = np.concatenate((phi, pho_aoa), axis = 1)
            steervec_u = np.exp(1j * angles * (D_u/CARRIER_WAVELENGTH))                            #uniform steering vectors
            symbols = np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) + 1j*np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) #QPSK symbols
            #distorted_symbols = np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, array_size)) + 1j*np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, array_size)) #QPSK symbols
            
            
            x_u = np.zeros(([NUMBER_OF_SENSORS, NUMBER_OF_SNAPSHOTS]), dtype=complex)
            x_2 = np.zeros(([NUMBER_OF_SENSORS, NUMBER_OF_SNAPSHOTS]), dtype=complex) 
     
        
            for j in range(number_of_correlatedsources):
               
                pho[j] = np.random.uniform()
                deltaphi[j] = np.random.uniform(-np.pi,np.pi)
    
                distorted_symbols[:,j] = pho[j]*np.exp(1j*deltaphi[j])*symbols[:,0]
            
            
            # if NUMBER_OF_SOURCES < number_of_correlatedsources:
            #     symbols = np.append(symbols,np.zeros([len(symbols),1]),1)
            # symbols = symbols + distorted_symbols
            # stack symbols and distorted, not add them
            
            symbols2 = np.concatenate((symbols, distorted_symbols), axis = 1)
            
            
            for i in range(NUMBER_OF_SNAPSHOTS):
                
                x_u[:,i] = np.sum(np.tile(symbols2[i,:],[NUMBER_OF_SENSORS, 1])*steervec_u,1)  #uniformly sampled data
                noise = NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1]) + 1j * NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1])  
                x_u = x_u + noise
                
           
    
            return x_u

    def generate_faded_signals(self, signal_dir_rad, sensor_position, NUMBER_OF_SENSORS, NOISE_VARIANCE, NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES, CARRIER_WAVELENGTH):
        phi = 2*math.pi*np.sin(np.tile(signal_dir_rad,[NUMBER_OF_SENSORS, 1])) 
        D_u = np.tile(sensor_position, [1, NUMBER_OF_SOURCES])
        steervec_u = np.exp(1j * phi * (D_u/CARRIER_WAVELENGTH))                               #uniform steering vectors
        symbols = np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) + 1j*np.sign(np.random.randn(NUMBER_OF_SNAPSHOTS, NUMBER_OF_SOURCES)) #QPSK symbols
        x_u = np.zeros(([NUMBER_OF_SENSORS, NUMBER_OF_SNAPSHOTS]), dtype=complex)
        for i in range(NUMBER_OF_SNAPSHOTS):
            x_u[:,i] = np.sum(np.tile(symbols[i,:],[NUMBER_OF_SENSORS, 1])*steervec_u,1)  #uniformly sampled data
            noise = NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1]) + 1j * NOISE_VARIANCE * np.random.randn(x_u.shape[0],x_u.shape[1])  
            x_u = x_u + noise
        return x_u
    
    def get_rootmusic_estimate(self, input_data, frequency, sources):        
        estimator = estimation.music.RootMUSIC1D(frequency)
        covariance_matrix = input_data.dot(self.hermitian(input_data))       
        resolved, estimates = estimator.estimate(covariance_matrix, sources, unit='deg')
        angle_estimate = estimates.locations
        return angle_estimate 
           
    def splitandweave_complex_input(self, input_real, input_imag, output_tensor, iteration): 
        """ splitandweave_complex_input
        Arguments:
            input_real (float)  : real part of the matrix (size: M * N * L)
            input_imag (float)  : imaginary part of the matrix (size: M * N * L)
        
        Returns:
            split and interweaved tensor for (size: N * 2L)
        """
        
        number_sensor = len(input_real)
        for i in range(iteration):
            p=0
            for k in range(number_sensor): #WAS L
                output_tensor[p,i] = input_real[k,i]  
                p=p+1
                output_tensor[p,i] = input_imag[k,i]
                p=p+1
        return output_tensor    
        

class AmpPhaseFeatureExtract(DataGen):
    def __init__(self):
        pass
        
    def get_parsed_amplitude(self, input_real,input_imag): #start here
        """ get_parsed_amplitude
        Arguments:
            input_real (float)  : real part of the matrix (size:  N * L)
            input_imag (float)  : imaginary part of the matrix (size:  N * 2L)

        Returns:
            computed magnitude marix of the complex input matrix (size:  N * L)
        """
        x_amp = np.sqrt(np.square(input_real)+np.square(input_imag))
        return x_amp
    
    def get_parsed_phase(self, input_real,input_imag):
        """ get_parsed_amplitude
        Arguments:
            input_real (float)  : real part of the matrix (size:  N * L)
            input_imag (float)  : imaginary part of the matrix (size:  N * 2L)

        Returns:
            computed phase marix of the complex input matrix (size:  N * L)
        """
        div = input_imag/input_real
        x_phase = np.arctan(div)
        return x_phase

class UpperTriFeatureExtract(DataGen):
    def __init__(self):
        pass
    
    def get_upper_tri(self, covmat,number_sensors):
        """ get_upper_tri
        Arguments:
            covmat  (float)            : covariance matrix
            number_sensors (int)       : number of sensors in the array

        Returns:
            right upper triangle of the square matix            
        """
        upper_tri = covmat[np.triu_indices(number_sensors)]
        return upper_tri
    
    
    def get_parse_upper_tri(self,norm_upperd_r,norm_upperd_i):
        """ parseinput
        Arguments:
            norm_upperd_r      (float)    : real part of the input data (square matrix)
            norm_upperd_i      (float)    : imaginary part of the input data (square matrix)
        
        Returns:
            upper triangle of the square matrix including the diagonal

        """
        output_shape = 2 * len(norm_upperd_r)
        output_vector = np.zeros(output_shape,)
        p=0
        for k in range(len(norm_upperd_i)):
            output_vector[p] = norm_upperd_r[k]  
            p=p+1
            output_vector[p] = norm_upperd_i[k]
            p=p+1
        return output_vector
    def get_parsed_image(self,norm_upperd_r,norm_upperd_i):
        """ parseinput
        Arguments:
            norm_upperd_r      (float)    : real part of the input data (square matrix)
            norm_upperd_i      (float)    : imaginary part of the input data (square matrix)
        
        Returns:
            upper triangle of the square matrix including the diagonal

        """
        output_shape = 2 * len(norm_upperd_r) * len(norm_upperd_r)
        output_vector = np.zeros(output_shape,)
        p=0
        for k in range(len(norm_upperd_i)):
            for k2 in range(len(norm_upperd_i)):
                
                output_vector[p] = norm_upperd_r[k][k2]      
                p=p+1
                output_vector[p] = norm_upperd_i[k][k2]
                p=p+1
        return output_vector
        
    def get_normalized_input(self, input_data):
        """ parseamplitude
        Arguments:
            None

        Returns:
            None

        """
        l2_norm = lalg.norm(input_data,2)
        norm_input = input_data/l2_norm
        return norm_input
    
    def get_off_uppertri(self, covmat, number_sensors):    # TODO: fix naming
        """ get_uppertri
        Arguments:
            covmat  (float)            : covariance matrix
            number_sensors (int)       : number of sensors in the array

        Returns:
            one off right upper triangle of the square matix (no diagonal)           
        """
        off_upper_tri = covmat[np.triu_indices(number_sensors, k=1)]
        return off_upper_tri

class DiscreteTargetSpectrum(DataGen):
    def __init__(self):
        pass
        
    def get_integize_target(self, target):
        """ get_integize_target
        Arguments:
            target  (float)   : degress of incoming signals

        Returns:
            integized target values            
        """
        target = [v*10 for v in target]
        return target
        
    def get_encoded_target(self, spectrum, target):
        """ get_encoded_target
        Arguments:
            target  (float)   : degress of incoming signals

        Returns:
            one hot encoded target vector with respect to the spectrum            
        """
        encoder = MultiLabelBinarizer()
        encoder.fit([spectrum])
        target = encoder.transform((target))
        return target


#    