#%% -*- coding: utf-8 -*-
"""
Compilation of all Uncertainty Models and Data Storage

@author: singaravelan
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys
import pickle
import copy
import types
import functools
from dataclasses import dataclass, field


import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as st

import tensorflow as tf
# tf.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.backend import eval

from matplotlib import gridspec
from matplotlib import pyplot as plt

tfd = tfp.distributions

#%% Utility functions

def copy_func(f):
    
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    
    return g



#%% Base Class Definitions

@dataclass
class TensorData:
    """
    Data Class for transfering data between different models and classes
    
    Important Attributes:
        
        - data [ pd.DataFrame ] : Unaltered copy of data file.
        
        - data_scaled [ pd.DataFrame ] : Copy of data file on which transformations are made.
        
        - feature_names [ list(string) ] : Signal names that should be inputs to the models.
        
        - target_names [ list(string) ] : Signal names that should be outputs of the models.
        
        - ranges [ dict ] : Dictionary containing permissible ranges of signals to scaled down and up.
        
        - is_scaled [ dict ] : Dictionary of booleans indincating whether a signal is scaled or not.
        
        - tensors [ dict ] : Dicitonary of important attributes to be used in the model.
    
    """
    
    ### Uncertainty Data ###
    data : pd.DataFrame = field(default_factory=pd.DataFrame)
    data_scaled : pd.DataFrame = field(default_factory=pd.DataFrame)
    
    feature_names : list = field(default_factory=list)
    target_names : list = field(default_factory=list)
    
    ranges : dict = field(default_factory=dict)
    is_scaled : dict = field(default_factory=dict)
    
    tensors : dict = field(default_factory=dict)



class UncertaintyData:
    """
    Contains methods for preprocessing data and defines important attributes.
    
    Important Attributes:
        
        - data [ pd.DataFrame ] : Unaltered copy of data file.
        
        - data_scaled [ pd.DataFrame ] : Copy of data file on which transformations are made.
        
        - feature_names [ list(string) ] : Signal names that should be inputs to the models.
        
        - target_names [ list(string) ] : Signal names that should be outputs of the models.
        
        - ranges [ dict ] : Dictionary containing permissible ranges of signals to scaled down and up.
        
        - is_scaled [ dict ] : Dictionary of booleans indincating whether a signal is scaled or not.
        
        - tensors [ dict ] : Dicitonary of important attributes to be used in the model : 
            
            - features [ tf.tensor ] : A tensor reprsenting features of models.
            - targets [ tf.tensor ] : A tensor representing targets of models.
            - is_windowed [ bool ] : Boolean indicating if features are windowed.
            - window_stride [ int ] : Interger indicating stride size of windowed features.
            - window_size [ int ] : Integer indicating window size of windowed features.
            - noise [ bool ] : Boolean indicating if noise has been applied on tensors
            
    """
    
    def __init__(self):
        
        self.data = pd.DataFrame()
        self.data_scaled = pd.DataFrame()
        
        self.feature_names = ['current', 'soc', 'temperature']
        self.target_names = ['voltage']
        
        self.ranges = {'voltage' : [300, 400],
                       'current': [1e-2, 400],
                       'temperature': [15, 35]}
        
        self.is_scaled = copy.deepcopy(self.ranges)
        self.is_scaled.update((key,False) for key in self.is_scaled)
        
        self.tensors = {'features' : None,
                        'targets' : None,
                        'is_windowed' : bool,
                        'window_stride': None,
                        'window_size': None,
                        'noise': bool}
        
 
        
    def _clean_data(self, data):
        """
        Cleans columns with feature and target names.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame object.

        Returns
        -------
        data : pd.DataFrame
            Cleaned DataFrame object.

        """
        
        data = copy.deepcopy(data)
        column_names = self.feature_names + self.target_names
        data.dropna(subset=column_names, inplace=True)
        
        return data
    
    
    
    def _add_datetime(self, data):
        """
        Adds datetime column with datetime objects based on timestamp.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame object.

        Returns
        -------
        data : pd.DataFrame
            Output DataFrame object with added datetime column.

        """
        
        data = copy.deepcopy(data)    
        if 'datetime' not in data.columns:
            data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')

        return data
    
    
    
    def _is_scaled(self, data, tol=1.5):
        """
        Checks if input features and target columns of input DataFrame object are scaled between [-tol, tol].

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame object.
        tol : float, optional
            Tolerance value of scaled signal. The default is 1.5.

        Returns
        -------
        is_scaled : dict
            Dictionary of booleans indincating whether a signal is scaled or not.

        """
        
        data = copy.deepcopy(data)
        is_scaled = {}
        
        for key, limits in self.ranges.items():
            try:
                if ((data[key].abs().max()+1e-2)) < tol:
                    is_scaled[key] = True
                
                else:
                    is_scaled[key] = False
            
            except KeyError:
                pass
            
        return is_scaled
    
    
    
    def _scale_range(self, data, is_scaled, scaler='down', force=False):
        """
        Scales features and targets in input DataFrame object based on is_scaled.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame object.
        is_scaled : dict
            Dictionary of booleans indincating whether a signal is scaled or not.
        scaler : string, optional
            String indicating whether to scale up or down. Can be 'up' or 'down'. The default is 'down'.
        force : bool, optional
            Boolean to force directional scaling operation. The default is False.

        Returns
        -------
        data : pd.DataFrame
            Output DataFrame object with scaled featues and targets.
        is_scaled : dict
            Dictionary of booleans indincating whether a signal is scaled or not.

        """
        
        data = copy.deepcopy(data)
        is_scaled = copy.deepcopy(is_scaled)
        
        for key, limits in self.ranges.items():
            try:
                if (scaler == 'down' and not is_scaled[key]) or force:
                    data[key] = (data[key] - limits[0])/(limits[1] - limits[0])
                    is_scaled[key] = True
                    
                elif (scaler == 'up' and is_scaled[key]) or force:
                    data[key] = data[key] * (limits[1] - limits[0]) + limits[0]
                    is_scaled[key] = False
                    
            except KeyError:
                pass
        
        return data, is_scaled
    
    
    
    def scale_data(self, scaler='up', force=False):
        """
        Wrapper to call internal function : _scale_range().
        Scales data_scaled up or down respectively, and updates is_scaled.
        
        Parameters
        ----------
        scaler : string, optional
            String indicating whether to scale up or down. Can be 'up' or 'down'. The default is 'up'.
        force : bool, optional
            Boolean to force directional scaling operation. The default is False.

        Returns
        -------
        None.

        """
        
        self.data_scaled, self.is_scaled = self._scale_range(self.data_scaled, self.is_scaled, scaler, force)
   
    
    
    def prepare_scaled_data(self, scaler='down', force=False):
        """
        Wrapper which cleans data, adds datetime and scales data and appends scaled data and is_scaled dictionary attributes.

        Parameters
        ----------
        scaler : string, optional
            String indicating whether to scale up or down. Can be 'up' or 'down'. The default is 'down'.
        force : bool, optional
            Boolean to force directional scaling operation. The default is False.

        Returns
        -------
        None.

        """
        
        data = copy.deepcopy(self.data)
        
        data = self._clean_data(data)
        data = self._add_datetime(data)
        
        is_scaled = self._is_scaled(data)
        
        self.data_scaled, self.is_scaled = self._scale_range(data, is_scaled, scaler, force)
        
    
    
    def add_data(self, data, features=None, targets=None, ranges=None):
        """
        Add data (pd.DataFrame object) to class to preprocess.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe object to preprocess.
        features : list[string], optional
            Signal names that should be inputs to the models.
        targets : list[string], optional
            Signal names that should be outputs to the models.
        ranges : dict, optional
            Dictionary containing permissible ranges of signals to scaled down and up.

        Returns
        -------
        None.

        """
        
        self.data = data
        
        if features:
            self.feature_names = features
        if targets:
            self.target_names = targets
        if ranges:
            self.ranges = ranges
        
    
    
    def add_tensor_data(self, tensor_data):
        """
        Loads TensorData objects attributes onto this class instance.
        
        Parameters
        ----------
        tensor_data : TensorData
            TensorData instance to load.

        Returns
        -------
        None.

        """
        
        tensor_data = copy.deepcopy(tensor_data) 
        
        for key in self.__dict__:
            setattr(self, key, getattr(tensor_data, key))


    
    def get_tensor_data(self, tensor_data=None):
        """
        Loads key attributes into given TensorData object or into new TensorData object.

        Parameters
        ----------
        tensor_data : TensorData, optional
            TensorData object/instance to load key attributes into. The default is None.

        Returns
        -------
        tensor_data : TensorData
            TensorData object/instance with key attributes inside it.

        """
        
        if not tensor_data:
            tensor_data = TensorData()
        
        for key in self.__dict__:
            setattr(tensor_data, key, getattr(self, key))
            
        return tensor_data
        
        
        
    def plot_data(self, signals=['voltage', 'current', 'soc', 'temperature'], scaler='up'):
        """
        Visualizes data_scaled DataFrame object.

        

        Parameters
        ----------
        signals : list[string], optional
            List of signal inputs to plot. The default is ['voltage', 'current', 'soc', 'temperature'].
        scaler : string, optional
            Argument whether to plot the scaled 'up' or 'down' version of data. The default is 'up'.

        Returns
        -------
        fig : matplotlib.figure
            Figure handle of figure.

        """
        
        data = copy.deepcopy(self.data_scaled)
        data, _ = self._scale_range(data, self.is_scaled, scaler=scaler)
        data.set_index('datetime', drop=True, inplace=True)
        
        colors = ['red', 'orange', 'green', 'black', 'blue', 'violet']
        
        fig = plt.figure()
        gs = gridspec.GridSpec(len(signals), 1, left=0.05, right=0.95)
        
        for i in range(len(signals)):
            
            ax = fig.add_subplot(gs[i, :])
            ax.plot(data[signals[i]], color=colors[i], label=signals[i])
            ax.legend()
        
        return fig
    
    
    
    def sample_uniform_tensors(self, sample_size=10000, overwrite=True):
        """
        Compression function to uniformly sample from training tensors and reduce size.
        To be used when training tensors are too large and training takes long.

        Parameters
        ----------
        sample_size : int, optional
            Sample size to reduce tensors size to. The default is 10000.
        overwrite : bool, optional
            Boolean to overwrite training feature and target tensors. The default is True.

        Returns
        -------
        tensor_features : tf.tensor
            Tensor of features to be used for training models.
        tensor_targets : tf.tensor
            Tensor of targets to be used for training models.

        """
        
        numpy_features = copy.deepcopy(self.tensors['features'].numpy())
        numpy_targets = copy.deepcopy(self.tensors['targets'].numpy())
        
        rand_index = np.random.randint(0,len(numpy_features),sample_size)
        
        numpy_features = numpy_features[rand_index]
        numpy_targets = numpy_targets[rand_index]
        
        tensor_features = tf.convert_to_tensor(numpy_features)
        tensor_targets = tf.convert_to_tensor(numpy_targets)
        
        if overwrite:
            self.tensors['features'] = tensor_features
            self.tensors['targets'] = tensor_targets
        
        return tensor_features, tensor_targets
            
    
    
    #--- Obselete ---#
    def compute_tensors(self, force=False):
        """
        Generates unwindowed tensors for training models.
        Updates tensors attribute.
        [ Currently Obselete ] 

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        
        self.prepare_scaled_data('down', force)
        # self.scale_data('down', force)
        
        data = copy.deepcopy(self.data_scaled)
        
        features = data[self.feature_names]
        tensor_features = tf.convert_to_tensor(features)
        
        targets = data[self.target_names]
        tensor_targets = tf.convert_to_tensor(targets)

        self.tensors = {'features' : tensor_features,
                        'targets' : tensor_targets,
                        'is_windowed' : False,
                        'window_stride': None,
                        'window_size': None,
                        'noise': False,
                        'clustering': False}



    def compute_tensors_windowed(self, stride=1, window_size=5, force=False, clustering_mode=False):
        """
        Generates windowed tensors for training models.
        Updates tensors attribute.
        

        Parameters
        ----------
        stride : int, optional
            Stride of tensor windows. The default is 1.
        window_size : int, optional
            Window size of tensor windows. The default is 5.
        force : bool, optional
            Boolean to force directional scaling operation. The default is False.
        clustering_mode : bool, optional
            Boolean to indicate whether the data is prepared for clustering or regression mode. The default is False.

        Returns
        -------
        None.

        """
        
        self.prepare_scaled_data('down', force)
        # self.scale_data('down',force)
        
        data = copy.deepcopy(self.data_scaled)
        
        features = data[self.feature_names]
        numpy_features = features.to_numpy()
        # as_strided(b_numpy, (2,10,3), (8*1,8,8*247239)) ---> 2 windows of size (10,3), with stride 1, of data type float64 (8 bytes)
        numpy_features_windowed = np.lib.stride_tricks.as_strided(x=numpy_features,
                                                                  shape=(((np.shape(numpy_features)[0]-window_size)//stride)+1, window_size, np.shape(numpy_features)[1]),
                                                                  strides=(8*stride, 8, 8*np.shape(numpy_features)[0]))
        
        tensor_features = tf.convert_to_tensor(numpy_features_windowed)
        
        
        targets = data[self.target_names]
        numpy_targets = targets.to_numpy()
        
        numpy_targets_windowed = np.lib.stride_tricks.as_strided(x=numpy_targets,
                                                                 shape=(((np.shape(numpy_targets)[0]-window_size)//stride)+1, window_size, np.shape(numpy_targets)[1]),
                                                                 strides=(8*stride, 8, 8*np.shape(numpy_targets)[0]))
        
        if not clustering_mode:
            numpy_targets_windowed = numpy_targets_windowed[:,-1,:]
        
        tensor_targets = tf.convert_to_tensor(numpy_targets_windowed)
        
        self.tensors = {'features' : tensor_features,
                        'targets' : tensor_targets,
                        'is_windowed' : True,
                        'window_stride': stride,
                        'window_size': window_size,
                        'noise': False,
                        'clustering': False}
        
        if clustering_mode:
            self.tensors['clustering'] = True
        
    
    
    def compute_tensors_windowed_noise(self, samples=100, noise_func=None, noise_std=1, use_signal='voltage', stride=1, window_size=20, force=False, clustering_mode=False):
        """
        Generates windowed tensors for training models with noise in specified signals.
        Updates tensors attribute.
        [ To be updated]

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame object.
        first : int, optional
            First index of data from which to be used. The default is 4000.
        last : int, optional
            Last index of data upto which to be used. The default is 7000.
        samples : int, optional
            Number of samples of data that needed to create noise. The default is 100.
        noise_func : func, optional
            Function to be passed according to which noises will be created on signal. The defaul is defined inside this function.
        use_signal : string, optional
            Signal string on which noise needs to be created. The default is 'voltage'.
        stride : int, optional
            Stride of tensor windows. The default is 1.
        window_size : int, optional
            Window size of tensor windows. The default is 5.
        force : bool, optional
            Boolean to force directional scaling operation. The default is False.
        clustering_mode : bool, optional
            Boolean to indicate whether the data is prepared for clustering or regression mode. The default is False.

        Returns
        -------
        data : pd.DataFrame
            Output DataFrame object with prepared featues and targets.
        is_scaled : dict
            Dictionary of booleans indincating whether a signal is scaled or not.
        tensor_features : tf.tensor
            Tensor of features (windowed and noised) to be used for training models.
        tensor_targets : tf.tensor
            Tensor of targets (noised) to be used for training models.

        """
        
        self.prepare_scaled_data('down', force)
        # self.scale_data('down',force)
        
        data = copy.deepcopy(self.data_scaled)
        
        
        if noise_func is None:
        
            def noise_func(df:pd.DataFrame, use_signal, stddev_scale=noise_std*1e-2):
                
                df_new = copy.deepcopy(df)
                
                mean = np.zeros(len(df))
                stddev = np.ones(len(df))*stddev_scale
                
                df_new[use_signal+'_clean'] = df[use_signal]
                df_new[use_signal] = df[use_signal] + np.random.normal(loc=mean, scale=stddev)
                df_new['mean_error'] = mean
                df_new['stddev_error'] = stddev
                
                return df_new
        
        
        for i in range(samples):
            
            data_noise = noise_func(data, use_signal)
            
            features = data_noise[self.feature_names]
            numpy_features = features.to_numpy()
            # as_strided(b_numpy, (2,10,3), (8*1,8,8*247239)) ---> 2 windows of size (10,3), with stride 1, of data type float64 (8 bytes)
            numpy_features_windowed = np.lib.stride_tricks.as_strided(x=numpy_features,
                                                                      shape=(((np.shape(numpy_features)[0]-window_size)//stride)+1, window_size, np.shape(numpy_features)[1]),
                                                                      strides=(8*stride, 8, 8*np.shape(numpy_features)[0]))
            
            tensor_features = tf.convert_to_tensor(numpy_features_windowed)
            
            
            targets = data_noise[self.target_names]
            numpy_targets = targets.to_numpy()
            
            numpy_targets_windowed = np.lib.stride_tricks.as_strided(x=numpy_targets,
                                                                     shape=(((np.shape(numpy_targets)[0]-window_size)//stride)+1, window_size, np.shape(numpy_targets)[1]),
                                                                     strides=(8*stride, 8, 8*np.shape(numpy_targets)[0]))
            
            if not clustering_mode:
                numpy_targets_windowed = numpy_targets_windowed[:,-1,:]
            
            tensor_targets = tf.convert_to_tensor(numpy_targets_windowed)
            
            
            if 'noise_features' not in vars():
                noise_features = tensor_features.numpy()
                noise_targets = tensor_targets.numpy()
                
            else:
                noise_features = np.concatenate((noise_features, tensor_features.numpy()), axis=0)
                noise_targets = np.concatenate((noise_targets, tensor_targets.numpy()), axis=0)
                
                
        tensor_features = tf.convert_to_tensor(noise_features)
        tensor_targets = tf.convert_to_tensor(noise_targets)
        
        
        self.tensors = {'features' : tensor_features,
                        'targets' : tensor_targets,
                        'is_windowed' : True,
                        'window_stride': stride,
                        'window_size': window_size,
                        'noise': True,
                        'clustering': False}
            
    
    
    #--- Work in Progress ---#
    def compute_binned_df(self, bins, intervals): 
        """
        Bins df_scaled into specified bins according to intervals.
        [To be implemented]

        Parameters
        ----------
        bins : TYPE
            DESCRIPTION.
        intervals : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        pass
    
    
        
    def reset(self):
        """
        Resets instance to default __init__ status.

        Returns
        -------
        None.

        """
        
        self.__init__()

    




class UncertaintyModel:
    """
    Acts as a base model class and contains common methods and attributes for various models.

    Important Attributes:
        
        - models [ dict ] : Dictionary of various model objects.
        
        - model_parameters [ dict ] : Dictionary of model parameters.
        
        - training_parameters [ dict ] : Dictionary of training parameters.
        
        - predicted_data [ pd.DataFrame ] : DataFrame object with predicted outputs.
        
        - history [ tf.keras.callbacks.History ] : Recent training history object.
        
        - report [ dict ] : Dicitionary with summarizing report.
        
        - Attributes of TensorData instance when a TensorData object is loaded.  
    
    """
    
    
    def __init__(self):
        
        self.models = {}
        
        self.model_parameters = None
        self.training_parameters = None
        
        self.predicted_data = None
        
        self.history = None
        
        self.report = None
        
        
        
    def save_model(self):
        pass
        
    def load_model(self):
        pass



    def add_tensor_data(self, tensor_data):
        """
        Adds/Loads information from TensorData instance.  
        
        Parameters
        ----------
        tensor_data : TensorData
            TensorData instance to load.
            
        Returns
        -------
        None.

        """

        tensor_data = copy.deepcopy(tensor_data) 
        
        for key in tensor_data.__dict__:
            setattr(self, key, getattr(tensor_data, key))



    def _Independent_loc_scale(self, kernel_size, bias_size=0, dtype=None, scale_offset=1e-3):
        """
        Independent Normal Distribution for Bayesian Node (variable mean and stddev).

        Parameters
        ----------
        kernel_size : TYPE
            DESCRIPTION.
        bias_size : TYPE, optional
            DESCRIPTION. The default is 0.
        dtype : TYPE, optional
            DESCRIPTION. The default is None.
        scale_offset : TYPE, optional
            DESCRIPTION. The default is 1e-3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        n = kernel_size + bias_size
        # c = np.log(np.expm1(1.))
        c = 0
        
        return keras.Sequential([
            tfp.layers.VariableLayer(2*n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t[..., :n], scale=scale_offset + tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1))
            ])
               
    
    
    def _Independent_loc(self, kernel_size, bias_size=0, dtype=None, scale_offset=1e-3):
        """
        Independent Normal Distribution for Bayesian Node (variable mean and fixed stddev).

        Parameters
        ----------
        kernel_size : TYPE
            DESCRIPTION.
        bias_size : TYPE, optional
            DESCRIPTION. The default is 0.
        dtype : TYPE, optional
            DESCRIPTION. The default is None.
        scale_offset : TYPE, optional
            DESCRIPTION. The default is 1e-3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        n = kernel_size + bias_size
        
        return keras.Sequential([
            tfp.layers.VariableLayer(n , dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=scale_offset), reinterpreted_batch_ndims=1))
            ])

    
    def _Independent_scale(self, kernel_size, bias_size=0, dtype=None, scale_offset=1e-3):
        """
        Independent Normal Distribution for Bayesian Node (fixed mean and variable stddev).

        Parameters
        ----------
        kernel_size : TYPE
            DESCRIPTION.
        bias_size : TYPE, optional
            DESCRIPTION. The default is 0.
        dtype : TYPE, optional
            DESCRIPTION. The default is None.
        scale_offset : TYPE, optional
            DESCRIPTION. The default is 1e-3.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        n = kernel_size + bias_size
        # c = np.log(np.expm1(1.))
        c = 0
        
        return keras.Sequential([
            tfp.layers.VariableLayer(n , dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=scale_offset + tf.nn.softplus(c + t)), reinterpreted_batch_ndims=1))
            ])
    
    
    def _Independent(self, kernel_size, bias_size=0, dtype=None, scale_offset=1):
        """
        Independent Normal Distribution for Bayesian Node (fixed mean and fixed stddev).

        Parameters
        ----------
        kernel_size : TYPE
            DESCRIPTION.
        bias_size : TYPE, optional
            DESCRIPTION. The default is 0.
        dtype : TYPE, optional
            DESCRIPTION. The default is None.
        scale_offset : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        n = kernel_size + bias_size
        
        return keras.Sequential([
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=scale_offset), reinterpreted_batch_ndims=1))
            ])   
    
    
    
    def add_bayesian_tail(self, layer=-3, config=None, model_name=None):
        """
        Addon Bayesian tail for model. Currently works only with CNN model.

        Parameters
        ----------
        layer : int , The default is -3.
            Layer number of parent model onto which to attach tail. 
        config : dict, optional
            Dictionary of configuration. The default is None.
        model_name : string
            Name of the model onto which to add tail. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            Creates a new model with tail and adds it to models attribute.

        """
        
        if not config:
            
            config = {
                
                'dense_units_bayesian' : [5, 3],
                'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                'activation_bayesian' : "relu",
                'distribution_output' : False
                
                }

        model = keras.models.clone_model(self.models[model_name])
        
        for layer in model.layers:
            layer.trainable = False
        
        features = model.layers[layer].output
        
        for units in config['dense_units_bayesian']:

            features = tfp.layers.DenseVariational(
                units = units,
                activation = config['activation_bayesian'],
                make_prior_fn = self._Independent_loc_scale,
                make_posterior_fn = self._Independent_loc_scale,
                kl_weight = 1/(self.tensors['features'][0]),
                activity_regularizer = config['activity_regularizer']
                )(features)
        
        outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                        activation=None)(features)
        
        if config['distribution_output']:
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
        
        model = keras.Model(inputs=model.input, outputs=outputs)
        model_name_new = model_name + '_bayesian_tail'
        
        self.models[model_name_new] = model
        
        return model
    
    
    
    def add_dropout_tail(self, layer=-3, config=None, model_name=None):
        """
        Addon Dropout tail for model. Currently works only with CNN model.

        Parameters
        ----------
        layer : int , The default is -3.
            Layer number of parent model onto which to attach tail.
        config : dict, optional
            Dictionary of configuration. The default is None.
        model_name : string
            Name of the model onto which to add tail. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            Creates a new model with tail and adds it to models attribute.

        """
        
        if not config:
            
            config = {
                
                'dense_units_dropout' : [30,30],
                'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                'dropout_rate' : 0.5,
                'activation_dropout' : "relu",
                'distribution_output' : False
                
                }

        model = keras.models.clone_model(self.models[model_name])
        
        for layer in model.layers:
            layer.trainable = False
        
        features = model.layers[layer].output
        
        for units in config['dense_units_dropout']:

            features = layers.Dense(
                units = units,
                activation = config['activation_dropout'],
                activity_regularizer = config['activity_regularizer']
                )(features)
            
            features = layers.Dropout(rate = config['dropout_rate'])(features, training=True)
        
        
        outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                        activation=None)(features)
        
        if config['distribution_output']:
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
        
        
        model = keras.Model(inputs=model.input, outputs=outputs)
        model_name_new = model_name + '_dropout_tail'
        
        self.models[model_name_new] = model
        
        return model
    
    
    
    def add_mixture_tail(self, layer=-4, config=None, model_name=None):
        """
        Addon Mixture tail for model. Currently works only with CNN model.

        Parameters
        ----------
        layer : int , The default is -4.
            Layer number of parent model onto which to attach tail.
        config : dict, optional
            Dictionary of configuration. The default is None.
        model_name : string
            Name of the model onto which to add tail. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a new model with tail and adds it to models attribute.

        """
        
        if not config:
            
            config = {
                
                'dense_units_mixture' : [20,20],
                'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                'activation_mixture' : "relu",
                'distribution_components' : 1,
                'distribution_output' : True
                
                }
        
        model = keras.models.clone_model(self.models[model_name])
        
        for layer in model.layers:
            layer.trainable = False
        
        features = model.layers[layer].output
        
        for units in config['dense_units_mixture']:

            features = layers.Dense(
                units = units,
                activation = config['activation_mixture'],
                activity_regularizer = config['activity_regularizer']
                )(features)
        
            
        params_size = tfp.layers.MixtureNormal.params_size(num_components=config['distribution_components'], event_shape=[self.tensors['targets'].shape[-1]])
        
        distribution_params = layers.Dense(params_size, activation=None)(features)
        outputs = tfp.layers.MixtureNormal(num_components=config['distribution_components'], event_shape=[self.tensors['targets'].shape[-1]])(distribution_params)
    
    
        model = keras.Model(inputs=model.input, outputs=outputs)
        model_name_new = model_name + '_mixture_tail'
        
        self.models[model_name_new] = model
        
        return model
    
    
    def _negative_loglikelihood(self, targets, estimated_distribution):
        """
        Negative log likelihood to use as loss while trianing probabilstic models.

        Parameters
        ----------
        targets : tfd.Distribution
            Target distribtuion.
        estimated_distribution : tfd.Distribution
            estimated distribtuion.
            
        Returns
        -------
        Negative log likelihood loss

        """
        
        return -estimated_distribution.log_prob(targets)
    
    
    
    def fit_model(self, training_parameters=None, model_name=None):
        """
        Generic function to fit all models.

        Parameters
        ----------
        training_parameters : dict, optional
            Dictionary of training parameters. The default is None.
        model_name : string
            Name of the model to train. The default is None.

        Returns
        -------
        tf.Keras.Hitsory.history 
            History object.

        """
        
        if not training_parameters:
            training_parameters = self.training_parameters
        
        model = self.models[model_name]
        
        if training_parameters['KL_loss']:
            loss = self._negative_loglikelihood
        else:
            loss = tf.keras.losses.MeanSquaredError()
        
        ### Callbacks ###
        callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                                   patience=20,
                                                                   verbose=0,
                                                                   mode='min')
        
        callback_terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
        
        callback_lr_plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor='root_mean_squared_error',
                                                                  factor=0.5,
                                                                  patience=15,
                                                                  verbose=0,
                                                                  mode='min',
                                                                  min_delta=0.001,
                                                                  cooldown=0,
                                                                  min_lr=0)
    
        callbacks = [callback_terminate_on_nan, callback_early_stopping]
        
        model.compile(
                optimizer = keras.optimizers.Adam(learning_rate=training_parameters['learning_rate']),
                loss = loss,
                metrics = [keras.metrics.RootMeanSquaredError()]
                )
        
        print("Start training the model...")
        self.history = model.fit(
            x = self.tensors['features'],
            y = self.tensors['targets'],
            batch_size = training_parameters['batch_size'],
            epochs = training_parameters['epochs'],
            verbose = training_parameters['verbose'],
            callbacks = callbacks,
            validation_split = training_parameters['validation_split'],
            shuffle = True
            )
        print("Model training finished.")
        
        self.models[model_name] = model
        
        return self.history
    
    
    
    def predict_on_data(self, test_df, model_name=None, samples=100, force_sampling=False, tol=3):
        """
        Predicts model on given pd.DataFrame object.

        Parameters
        ----------
        test_df : pd.DataFrame
            Datadrame object on which prediction should be made.
        model_name : string
            Name of the model to use to predict. The default is None.
        samples : int, optional
            Manually sample by predicting the output. The default is 100.
        force_sampling : bool, optional
            Boolean to toggle force manual sampling. The default is False.
        tol : float, optional
            Tolerance value for flagging possible Faults. The default is 3.

        Returns
        -------
        predicted_data : pd.DataFrame
            Predicted Dataframe with predicted value with bounds.

        """
        
        uncertainty_data = UncertaintyData()
        uncertainty_data.add_data(test_df, self.feature_names, self.target_names, self.ranges)
        
        uncertainty_data.compute_tensors_windowed(stride=1, window_size=self.tensors['window_size'])
    
        test_df_scaled = uncertainty_data.data_scaled
        test_features = uncertainty_data.tensors['features']
        test_targets = uncertainty_data.tensors['targets']
        
        uncertainty_data.scale_data('up')
        test_df = uncertainty_data.data_scaled
        
        
        predicted_data = pd.DataFrame(index=test_df_scaled['datetime'])
        
        model = self.models[model_name]
        
        
        if not force_sampling and isinstance(model.layers[-1], tfp.layers.DistributionLambda):
            mean = model(test_features).mean().numpy().squeeze()
            stddev = model(test_features).stddev().numpy().squeeze()
            
        else:
            prediction = np.empty([(len(test_df_scaled)-self.tensors['window_size']+1),samples])
            
            for i in range(samples):
                prediction[:,i] = model(test_features).numpy().squeeze()
            
            mean = np.mean(prediction, axis=1)
            stddev = np.std(prediction, axis=1)
        
        
        test_target_data = np.vstack((mean,
                                      stddev,
                                      mean - tol*stddev,
                                      mean + tol*stddev,
                                      test_targets.numpy().squeeze())).T
        
        test_target_data = np.pad(test_target_data,((self.tensors['window_size']-1, 0),(0, 0)), mode='constant', constant_values=np.NaN)
        
        predicted_data[['mean','stddev','lower','upper','actual']] = test_target_data
        
        limits = self.ranges[self.target_names[0]]
        predicted_data = predicted_data * (limits[1] - limits[0]) + limits[0]
        predicted_data['stddev'] -= limits[0]
        
        test_df.set_index('datetime',inplace=True)
        for i in self.feature_names: predicted_data[i] = test_df[i]
        
        predicted_data['fault'] = False
        
        predicted_data.loc[
            (predicted_data['actual'] < predicted_data['lower']) | 
            (predicted_data['actual'] > predicted_data['upper']), 
            'fault'] = True
        
        predicted_data['ideal_deviation_score'] = np.abs(np.random.normal(0, 1, len(predicted_data)))
        predicted_data['deviation_score'] = np.abs(predicted_data['actual'] - predicted_data['mean']) / predicted_data['stddev']
        predicted_data['stddev_data'] = np.abs(predicted_data['actual'] - predicted_data['mean'])
        
        self.predicted_data = predicted_data
        
        return predicted_data




    def _compute_distribution_measures(self, ):
        
        predicted_data = copy.deepcopy(self.predicted_data)        
        
        

    #--- Obselete ---#
    def create_report(self):
        """
        Obselete , will be updated.

        Returns
        -------
        report : TYPE
            DESCRIPTION.

        """
        
        data, _ = self._scale_range(self.data_scaled, self.is_scaled, scaler='up')
        data.set_index('datetime',inplace=True)
        predicted_data = copy.deepcopy(self.predicted_data)
        
        self.report['mean_MAE'] = np.mean(np.abs(data['voltage'] - predicted_data['mean']))
        self.report['mean_RMSE'] = np.sqrt(np.mean((data['voltage'] - predicted_data['mean'])**2))
        
        self.report['stddev_MAE'] = np.mean(np.abs(data['stddev_error'] - predicted_data['stddev']))
        self.report['stddev_RMSE'] = np.sqrt(np.mean((data['stddev_error'] - predicted_data['stddev'])**2))
        
        self.report['training_data'] = data
        self.report['test_data'] = predicted_data
        
        report = {**self.model_parameters, **self.training_parameters, **self.report}
        
        return report

    

    def plot_summary(self, plot_fault=False, plot_legend=True, loc=1):
        """
        Plots a summary of prediced_data

        Parameters
        ----------
        plot_fault : bool, optional
            Boolean toggle to highlight possible faults. The default is False.

        Returns
        -------
        fig : matplotlib.figure
            Figure handle of figure.

        """
        data = copy.deepcopy(self.predicted_data)
        
        colors = ['red', 'orange', 'green', 'black', 'blue', 'violet']
        
        fig = plt.figure()
        fig.suptitle(" Summary Statistics ")
        
        # fig.suptitle("   window size:" + str(self.tensors['window_size']) +
        #              "   stride size:" + str(self.tensors['window_stride']) + "\n" + 
        #             str(self.model_parameters) + "\n" +  
        #             str(self.training_parameters) + "\n")
        
        rows = len(self.feature_names) + len(self.target_names) + 1
        
        gs = gridspec.GridSpec(rows, 5, left=0.05, right=0.95)
        
        row_i = 0
        
        # ax1 = fig.add_subplot(gs[0, :])
        # ax1.plot(data['datetime'], data[self.target_names], label=self.target_names)
        # # data['lower'] = data[self.target_names[0]] - 3*data['stddev_error']
        # # data['upper'] = data[self.target_names[0]] + 3*data['stddev_error']
        # # ax1.fill_between(data['datetime'], 
        # #                  data['lower'], 
        # #                  data['upper'], 
        # #                  color = 'red',
        # #                  alpha = 0.2,
        # #                  label = "Training Data Noise (3 std dev [99.7 %])")
        # ax1.set_title('Training Data')
        # ax1.legend()
        
        for features in self.feature_names:
            
            ax = fig.add_subplot(gs[row_i, :])
            ax.plot(data.index, data[features], color=colors[row_i] , label=features)
            # ax.set_title('Test Data - ' + features)
            if plot_legend:
                ax.legend(loc=loc)
        
            row_i = row_i + 1
            
        
        ax = fig.add_subplot(gs[row_i, :])
        ax.plot(data.index, data['actual'], color='blue', label="Actual")
        ax.plot(data.index, data['mean'], color='black', label="Predicted Mean")
        ax.fill_between(data.index, 
                        data['lower'], 
                        data['upper'], 
                        color = 'red',
                        alpha = 0.2,
                        label = "Confidence Interval (3 std dev [99.7 %])")
        if plot_fault:
            fault_df = data[data['fault'] == True]
            ax.plot(fault_df.index, fault_df['actual'], 'rX', markersize=0.5, label='Predicted Fault')
        
        # ax.set_title('Model Prediciton on Test Data')
        if plot_legend:
            ax.legend(loc=loc)
        
        row_i = row_i + 1
        
        
        ax = fig.add_subplot(gs[row_i, :])
        ax.vlines(data.index, 0.0 ,data['deviation_score'], label='Deviation Score')
        # ax.set_title('Deviation Score')
        if plot_legend:
            ax.legend(loc=loc)
        
        
        # ax4 = fig.add_subplot(gs[2, 2])
        # ax4.plot(self.history.history['loss'], label="Negative Log Likelihood")
        # ax4.set_title('Training History')
        # ax4.legend()
        
        return fig
    
    
    

#%% Models

class ExpModel(UncertaintyModel):
    """
    Experimental Model V -> SUM( (DNN(I, SOC, T)-> Corr_Matrix) @ (I, SOC, T).T )
    
    """
    
    def __init__(self, **kwargs):
        
        super(ExpModel, self).__init__(**kwargs)
        
        self.model_parameters = {'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : False,
                                    'batch_size' : 128,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        
        features = layers.Dense(units = 18, activation = "relu")(inputs[..., -1, :])
        features = layers.Dense(units = 18, activation = "relu")(features)
        features = layers.Dense(units=9, activation=None)(features)
        
        correlation_matrix = layers.Reshape(target_shape=(3,3))(features)
        state_vector = layers.Reshape(target_shape=(3,1))(inputs[..., -1, :]) 
        
        mean = tf.reduce_sum(tf.matmul(correlation_matrix, state_vector), axis=1, keepdims=True)
        # stddev = tf.constant([1e-3], dtype=tf.float64)
        
        # outputs = layers.Concatenate(axis=1)([mean, stddev]) 
        # outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        model = keras.Model(inputs=inputs, outputs=mean)
        
        self.models['ExpModel'] = model
        
        return model
    
    
    
class ExpModel2(UncertaintyModel):
    """
    Experimental Model V -> SUM( (DNN(I, SOC, T)-> Corr_Matrix) @ (I, SOC, T).T )
    
    """
    
    def __init__(self, **kwargs):
        
        super(ExpModel2, self).__init__(**kwargs)
        
        self.model_parameters = {'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : False,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        

        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        
        features = self.inputs
        
        class BayesianDense(keras.layers.Layer):
            
            def __init__(self, prior, posterior, kl_weight):
                super(BayesianDense, self).__init__()
                
                self.dense_1 = layers.Dense(5, activation="relu")
                self.dense_2 = layers.Dense(5, activation="relu")
                self.dense_var =  tfp.layers.DenseVariational(1, activation=None,
                                                              make_prior_fn=prior,
                                                              make_posterior_fn=posterior,
                                                              kl_weight=kl_weight)
                
            def call(self, inputs):
                x = self.dense_1(inputs)
                x = self.dense_2(x)
                x = self.dense_var(x)
                
                return x
        
        prior = self._Independent
        posterior = self._Independent_loc_scale        
        kl_weight = 1/(self.tensors['features'][0]) 
        
        state = layers.Flatten()(features[..., -1, :])
        V_state = BayesianDense(prior=prior, posterior=posterior, kl_weight=kl_weight)(state)
        
        V_std = layers.Concatenate(axis=1)([state, V_state])
        V_std = layers.Dense(3, activation="relu")(V_std)
        V_std = layers.Dense(1, activation=None)(V_std)
        
        outputs_1 = layers.Concatenate(axis=1)([V_state, V_std])
        
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs_1)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(V_state)
        
        model = self.update_model()
        
        return model
    
    
    
    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['ExpModel2'] = model
        
        return model
    
    
    
         
    

class LinRDynOCV(UncertaintyModel):
    """
    Experimental Model V -> SUM( (DNN(I, SOC, T)-> Corr_Matrix) @ (I, SOC, T).T )
    
    """
    
    def __init__(self, **kwargs):
        
        super(LinRDynOCV, self).__init__(**kwargs)
        
        self.model_parameters = {}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 100,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        
        features = self.inputs
        
        class LinearR(tf.keras.layers.Layer):
            
            def __init__(self):
                super(LinearR, self).__init__()
            
            def build(self, input_shape):
               self.w_train = self.add_weight(name='Linear_R', shape=(1,), initializer="zeros", trainable=True)
            
            def call(self, inputs):
                return tf.multiply(inputs, self.w_train)
        
        current = layers.Flatten()(features[..., -1, 0])
        V_R = LinearR()(current)
        
        
        class DynOCV(keras.layers.Layer):
            
            def __init__(self):
                super(DynOCV, self).__init__()
                
                self.dense_1 = layers.Dense(5, activation="relu")
                self.dense_2 = layers.Dense(5, activation="relu")
                self.dense_3 = layers.Dense(1, activation=None)
        
            def call(self, inputs):
                x = self.dense_1(inputs)
                x = self.dense_2(x)
                x = self.dense_3(x)
                
                return x
        
        soc = layers.Flatten()(features[..., -1, 1])
        V_SOC = DynOCV()(soc)
        
        V_mean = layers.Add()([V_R, V_SOC])
        
        ### Output Layer ###

        V_std = layers.Flatten()(features[..., -1, :])
        V_std = layers.Concatenate(axis=1)([V_mean, V_std])
        V_std = layers.Dense(5, activation="relu")(V_std)
        V_std = layers.Dense(1, activation=None)(V_std)
        
        outputs_1 = layers.Concatenate(axis=1)([V_mean, V_std])         
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs_1)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(V_mean)
        
        model = self.update_model()
        
        return model



    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['LinRDynOCV'] = model
        
        return model
    
    


class PhysicsEnsemble(UncertaintyModel):
    """
    Experimental Model V -> SUM( (DNN(I, SOC, T)-> Corr_Matrix) @ (I, SOC, T).T )
    
    """
    
    def __init__(self, **kwargs):
        
        super(PhysicsEnsemble, self).__init__(**kwargs)
        
        self.model_parameters = {'use_approx_cap' : False}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 100,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                          self.tensors['features'].shape[2]))
        
        
        current = layers.Flatten()(self.inputs[..., -1, 0])
        current_vec = tf.expand_dims(self.inputs[..., 0], axis=-1)
        current_mean =  layers.Flatten()(tf.math.reduce_mean(self.inputs[..., 0], axis=1))
        current_std =  layers.Flatten()(tf.math.reduce_std(self.inputs[..., 0], axis=1))
        
        soc = layers.Flatten()(self.inputs[..., -1, 1])
        
        temp = layers.Flatten()(self.inputs[..., -1, 2])
        
        
        class LinearR(tf.keras.layers.Layer):
            
           def __init__(self, **kwargs):
               super(LinearR, self).__init__(**kwargs)
               self.dense = layers.Dense(1, activation="linear", use_bias=False)
       
           def call(self, inputs):
               x = self.dense(inputs)
               
               return x

        voltage_R = LinearR()(current)
        
        
        class DynC(tf.keras.layers.Layer):
            
            def __init__(self, **kwargs):
                super(DynC, self).__init__(**kwargs)
                
                # self.dense_1 = layers.Dense(5, activation="linear")
                self.rnn_1 = layers.LSTM(5, activation="relu", return_sequences=False)
                self.dense_2 = layers.Dense(5, activation="linear")
                self.dense_3 = layers.Dense(1, activation=None)
                
                
        
            def call(self, inputs):
                x = self.rnn_1(inputs)
                x = self.dense_2(x)
                x = self.dense_3(x)
                
                return x
        
        voltage_C = DynC()(current_vec)
        
        
        class LinearC(tf.keras.layers.Layer):
            
            def __init__(self, **kwargs):
                super(LinearC, self).__init__(**kwargs)
                self.dense = layers.Dense(1, activation="linear", use_bias=False)
        
            def call(self, inputs):
                x = self.dense(inputs)
                
                return x
        
        voltage_lin_C = LinearC()(current_mean)
        
        
        class DynOCV(keras.layers.Layer):
            
            def __init__(self, **kwargs):
                super(DynOCV, self).__init__(**kwargs)
                
                self.dense_1 = layers.Dense(5, activation="linear")
                self.dense_2 = layers.Dense(5, activation="linear")
                self.dense_3 = layers.Dense(1, activation=None)
        
            def call(self, inputs):
                x = self.dense_1(inputs)
                x = self.dense_2(x)
                x = self.dense_3(x)
                
                return x
            
        voltage_OCV = DynOCV()(soc)
        
        
        class DynTCV(keras.layers.Layer):
            
            def __init__(self, **kwargs):
                super(DynTCV, self).__init__(**kwargs)
                
                self.dense_1 = layers.Dense(5, activation="linear")
                self.dense_2 = layers.Dense(5, activation="linear")
                self.dense_3 = layers.Dense(1, activation=None)
        
            def call(self, inputs):
                x = self.dense_1(inputs)
                x = self.dense_2(x)
                x = self.dense_3(x)
                
                return x
        
        voltage_TCV = DynTCV()(temp)
        
        
        voltage_R_OCV_T = layers.Add()([voltage_R, voltage_OCV, voltage_TCV])
        
        ### adding polarization voltage drop ###
        if model_parameters['use_approx_cap']:
            voltage_R_OCV_T = layers.Add()([voltage_R_OCV_T, voltage_C])
        
        else:
            voltage_R_OCV_T = layers.Add()([voltage_R_OCV_T, voltage_lin_C])
        
        
        ### Output Layer ###
        class OutputUncertainty(tf.keras.layers.Layer):
            
            def __init__(self, **kwargs):
                super(OutputUncertainty, self).__init__(**kwargs)
                
                self.dense_1 = layers.Dense(5, activation="relu")
                self.dense_2 = layers.Dense(1, activation=None)
                
                
            def call(self, inputs):
                x = self.dense_1(inputs)
                x = self.dense_2(inputs)

                return x
        
        input_std = layers.Concatenate(axis=1)([current, soc, temp, voltage_R_OCV_T])
        voltage_std = OutputUncertainty()(input_std)
        
        outputs_1 = layers.Concatenate(axis=1)([voltage_R_OCV_T, voltage_std])
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs_1)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(voltage_R_OCV_T)
        
        model = self.update_model()
        
        return model    
        
    
    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['PhysicsEnsemble'] = model
        
        return model
    
    

class MLR(UncertaintyModel):
    """
    Multiple Linear Regession model Y[i] = A X[i] + B .
    
    """
    
    def __init__(self, **kwargs):
        
        super(MLR, self).__init__(**kwargs)
        
        self.model_parameters = {'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = inputs
        
        class StateMatrixMean(tf.keras.layers.Layer):
            
            def __init__(self):
                super(StateMatrixMean, self).__init__()
            
            def build(self, input_shape):
               self.w_train = self.add_weight(name='state_matrix_mean_w', shape=(3,1), initializer="ones", trainable=True)
               self.b_train = self.add_weight(name='state_matrix_mean_b', shape=(1,1), initializer="zeros", trainable=True)
            
            def call(self, inputs):
                return tf.tensordot(inputs[..., -1, :], self.w_train, 1) + self.b_train
        
        mean = StateMatrixMean()(inputs)
        
        
        class StateMatrixStddev(tf.keras.layers.Layer):
            
            def __init__(self):
                super(StateMatrixStddev, self).__init__()
            
            def build(self, input_shape):
               self.w_train = self.add_weight(name='state_matrix_stddev_w', shape=(3,1), initializer="zeros", trainable=True)
               self.b_train = self.add_weight(name='state_matrix_stddev_b', shape=(1,1), initializer="zeros", trainable=True)
            
            def call(self, inputs):
                return tf.tensordot(inputs[..., -1, :], self.w_train, 1) + self.b_train
        
        
        stddev = StateMatrixStddev()(inputs)
        
        if model_parameters['distribution_output']:
            
            outputs = layers.Concatenate(axis=1)([mean, stddev]) 
            outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        else:
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(mean)
        
        
        outputs = layers.Concatenate(axis=1)([mean, stddev]) 
    
        outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.models['MLR'] = model
        
        return model
        
        

class DNN(UncertaintyModel):
    """
    Simple Deep Neural Network Model.
    
    """
    
    def __init__(self, **kwargs):
        
        super(DNN, self).__init__(**kwargs)
        
        self.model_parameters = {'dense_units_vanilla' : [10,10],
                                 'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                                 'activation_vanilla' : "relu",
                                 'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.
            
        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = inputs
        
        
        ### Layer which only take last state ###
        class States(tf.keras.layers.Layer):
            
            def __init__(self):
                super(States, self).__init__()
            
            def call(self, inputs):
                return inputs[..., -1, :]
        
        features = States()(inputs)
        features = layers.Flatten()(features)
        
        
        ### Dense Layer ###
        if model_parameters['dense_units_vanilla']:
            for units in model_parameters['dense_units_vanilla']:
                features = layers.Dense(
                    units = units,
                    activation = model_parameters['activation_vanilla'],
                    activity_regularizer = model_parameters['activity_regularizer']
                    )(features)
    
    
        ### Output Layer ###
        if model_parameters['distribution_output']:
            outputs = layers.Dense(units=2*(self.tensors['targets'].shape[-1]),
                                   activation=None)(features)
            
            outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        else:
            outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                                   activation=None)(features)
            
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
        
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.models['DNN'] = model
        
        return model



class RNN(UncertaintyModel):
    """
    Recurrent Neural Network - SimpleRNN or GRU or LSTM
    
    """
    def __init__(self, **kwargs):
        
        super(RNN, self).__init__(**kwargs)
        
        self.model_parameters = {'RNN_type': 'LSTM',
                                 'RNN_units' : 10,
                                 'RNN_activation' : "relu",
                                 'RNN_dropout' : 0.0,
                                 'dense_units_vanilla' : [10,10],
                                 'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                                 'activation_vanilla' : "relu",
                                 'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
        
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = inputs
        
        ### RNN Layer ###
        if model_parameters['RNN_type'] == 'SimpleRNN':
            RNN_layer = layers.SimpleRNN
        
        elif model_parameters['RNN_type'] == 'GRU':
            RNN_layer = layers.GRU
            
        elif model_parameters['RNN_type'] == 'LSTM':
            RNN_layer = layers.LSTM
        
        features = RNN_layer(
            units = model_parameters['RNN_units'],
            activation = model_parameters['RNN_activation'],
            activity_regularizer = model_parameters['activity_regularizer'],
            dropout = model_parameters['RNN_dropout'],
            return_sequences = False
            )(features)
        
        
        ### Dense Layer ###
        if model_parameters['dense_units_vanilla']:
            for units in model_parameters['dense_units_vanilla']:
                features = layers.Dense(
                    units = units,
                    activation = model_parameters['activation_vanilla'],
                    activity_regularizer = model_parameters['activity_regularizer']
                    )(features)
    
    
        ### Output Layer ###
        if model_parameters['distribution_output']:
            outputs = layers.Dense(units=2*(self.tensors['targets'].shape[-1]),
                                    activation=None)(features)
            
            outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        else:
            outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                                   activation=None)(features)
            
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
            
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.models['RNN'] = model
        
        return model



class CNN(UncertaintyModel):
    """
    Convolution Neural Network model which takes in windowed inputs.
    
    """
    def __init__(self, **kwargs):
        
        super(CNN, self).__init__(**kwargs)
        
        self.model_parameters = {'filters' : 10,
                                 'kernel_size' : 3,
                                 'padding' : "causal",
                                 'activation_conv' : "relu",
                                 'dense_units_vanilla' : [10,10],
                                 'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                                 'activation_vanilla' : "relu",
                                 'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = inputs
        
        ### Conv Layer ###
        conv_layer = tf.keras.layers.Conv1D
        
        features = conv_layer(filters = model_parameters['filters'],
                              kernel_size = model_parameters['kernel_size'],
                              padding = model_parameters['padding'],
                              activation = model_parameters['activation_conv'],
                              activity_regularizer = model_parameters['activity_regularizer']
                              )(features)

        features = layers.Flatten()(features)
        
        ### Dense Layer ###
        if model_parameters['dense_units_vanilla']:
            for units in model_parameters['dense_units_vanilla']:
                features = layers.Dense(
                    units = units,
                    activation = model_parameters['activation_vanilla'],
                    activity_regularizer = model_parameters['activity_regularizer']
                    )(features)
    
    
        ### Output Layer ###
        if model_parameters['distribution_output']:
            outputs = layers.Dense(units=2*(self.tensors['targets'].shape[-1]),
                                   activation=None)(features)
            
            outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        else:
            outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                                   activation=None)(features)
            
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
        
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.models['CNN'] = model
        
        return model
    



class MultiHead(UncertaintyModel):
    """
    Multi Headed Attention inspired model.
    - head 1 (Current Standard Deviation in Window)
    - head 2 (Current Rate [Backward Difference] / Mean in Window)
    - head 3 (Weighted States with Offsets)
    
    """
    def __init__(self, **kwargs):
        
        super(MultiHead, self).__init__(**kwargs)
        
        self.model_parameters = {'use_current_rate' : False,
                                 'dense_units_vanilla' : [10,10],
                                 'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                                 'activation_vanilla' : "relu",
                                 'distribution_output' : True}

        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 200,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
    
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))

        
        
        class CurrentStdDev(tf.keras.layers.Layer):
            
            def __init__(self):
                super(CurrentStdDev, self).__init__()
            
            def build(self, input_shape):
                self.w_train = self.add_weight(name='current_std_dev_w', shape=(1,), initializer="ones", trainable=True)
                
            def call(self, inputs):
                return tf.math.reduce_std(inputs[..., 0], axis=1) * self.w_train
                
        features1 = CurrentStdDev()(inputs)
        features1 = layers.Flatten()(features1)
        
        if model_parameters['use_current_rate']:
            class CurrentRate(tf.keras.layers.Layer):
                
                def __init__(self):
                    super(CurrentRate, self).__init__()
                    
                def build(self, input_shape):
                    self.w_static = tf.constant([3./12., -16./12., 36./12., -48./12., 25./12.], dtype=tf.float32)
                    self.w_train = self.add_weight(name='current_rate_w', shape=(1,), initializer="ones", trainable=True)
                    
                def call(self, inputs):
                    return tf.math.reduce_sum(tf.multiply(inputs[..., 0], self.w_static), axis=1) * self.w_train
                
            features2 = CurrentRate()(inputs)
            features2 = layers.Flatten()(features2)
            
        else:
            class CurrentMean(tf.keras.layers.Layer):
            
                def __init__(self):
                    super(CurrentMean, self).__init__()
                
                def build(self, input_shape):
                    self.w_train = self.add_weight(name='current_mean_w', shape=(1,), initializer="ones", trainable=True)
                    
                def call(self, inputs):
                    return tf.math.reduce_mean(inputs[..., 0], axis=1) * self.w_train
                    
            features2 = CurrentMean()(inputs)
            features2 = layers.Flatten()(features2)
        
        
        class WeightedStates(tf.keras.layers.Layer):
            
            def __init__(self):
                super(WeightedStates, self).__init__()
            
            def build(self, input_shape):
               self.w_train = self.add_weight(name='weighted_states_w', shape=(1,3), initializer="ones", trainable=True)
               self.b_train = self.add_weight(name='weighted_states_b', shape=(1,3), initializer="zeros", trainable=True)
            
            def call(self, inputs):
                return tf.multiply(inputs[..., -1, :], self.w_train) + self.b_train
        
        features3 = WeightedStates()(inputs)
        features3 = layers.Flatten()(features3)
        
        features = layers.Concatenate(axis=1)([features1, features2, features3]) 
        
        
        
        ### Dense Layer ###
        if model_parameters['dense_units_vanilla']:
            for units in model_parameters['dense_units_vanilla']:
                features = layers.Dense(
                    units = units,
                    activation = model_parameters['activation_vanilla'],
                    activity_regularizer = model_parameters['activity_regularizer']
                    )(features)
    
    
        ### Output Layer ###
        if model_parameters['distribution_output']:
            outputs = layers.Dense(units=2*(self.tensors['targets'].shape[-1]),
                                   activation=None)(features)
            
            outputs = tfp.layers.IndependentNormal(1)(outputs)
        
        else:
            outputs = layers.Dense(units=self.tensors['targets'].shape[-1],
                                   activation=None)(features)
            
            outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs)
        
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.models['MultiHead'] = model
        
        return model
        


class DualChannelRNNHead(UncertaintyModel):
    """
    Recurrent Neural Network (SimpleRNN or GRU or LSTM) head with dual channel body
    - channel1 - Can be Epistemic / Deterministic - predicts mean
    - channel2 - Can be Epistemic / Deterministic - predicts stddev
    
    """
    def __init__(self, **kwargs):
        
        super(DualChannelRNNHead, self).__init__(**kwargs)
        
        self.model_parameters = {'RNN_type': 'LSTM',
                                 'RNN_units' : 10,
                                 'RNN_activation' : "relu",
                                 'RNN_dropout' : 0.0,
                                 'activity_regularizer' : tf.keras.regularizers.L2(1e-5),
                                 'channel1' : {'dense_units_vanilla' : [10,10],
                                               'activity_regularizer_vanilla' : tf.keras.regularizers.L2(1e-5),
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [1],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None},
                                 'channel2' : {'dense_units_vanilla' : [5],
                                               'activity_regularizer_vanilla' : None,
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None}
                                 }
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 100,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
        
        
        
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
        
        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = self.inputs
        
        ### RNN Layer ###
        if model_parameters['RNN_type'] == 'SimpleRNN':
            RNN_layer = layers.SimpleRNN
        
        elif model_parameters['RNN_type'] == 'GRU':
            RNN_layer = layers.GRU
            
        elif model_parameters['RNN_type'] == 'LSTM':
            RNN_layer = layers.LSTM
        
        features = RNN_layer(
            units = model_parameters['RNN_units'],
            activation = model_parameters['RNN_activation'],
            activity_regularizer = model_parameters['activity_regularizer'],
            dropout = model_parameters['RNN_dropout'],
            return_sequences = False
            )(features)
        
        
        prior = self._Independent
        posterior = self._Independent_loc_scale
        
        
        def channel_1(features, model_parameters=model_parameters['channel1']):
                
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla']
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
        
        
        def channel_2(features, model_parameters=model_parameters['channel2']):
                
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla']
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
            
        
        outputs1 = channel_1(features)
        outputs2 = channel_2(features = layers.Concatenate(axis=1)([features, outputs1]))
        
        outputs = layers.Concatenate(axis=1)([outputs1, outputs2]) 
        
        
        ### Output Layer ###
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs1)
            
        model = self.update_model()
        
        return model        



    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['DualChannelRNNHead'] = model
        
        return model



class DualChannelCNNHead(UncertaintyModel):
    """
    Convolution Neural Network head with dual channel body.
    - channel1 - Can be Epistemic / Deterministic - predicts mean
    - channel2 - Can be Epistemic / Deterministic - predicts stddev
    
    """
    
    def __init__(self, **kwargs):
    
        super(DualChannelCNNHead, self).__init__(**kwargs)
        
        self.model_parameters = {'filters' : 10,
                                 'kernel_size' : 2,
                                 'padding' : "causal",
                                 'activation_conv' : "relu",
                                 'activity_regularizer_conv' : tf.keras.regularizers.L2(1e-5),
                                 'channel1' : {'dense_units_vanilla' : [10,10],
                                               'activity_regularizer_vanilla' : tf.keras.regularizers.L2(1e-5),
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [1],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None},
                                 'channel2' : {'dense_units_vanilla' : [5],
                                               'activity_regularizer_vanilla' : None,
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None}
                                 }
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 100,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
    
    
    
    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        features = self.inputs
    
        ### Conv Layer ###
        conv_layer = tf.keras.layers.Conv1D
        
        features = conv_layer(filters = model_parameters['filters'],
                              kernel_size = model_parameters['kernel_size'],
                              padding = model_parameters['padding'],
                              activation = model_parameters['activation_conv'],
                              activity_regularizer = model_parameters['activity_regularizer_conv']
                              )(features)

        features = layers.Flatten()(features)
        
        
        prior = self._Independent
        posterior = self._Independent_loc_scale
        
        
        def channel_1(features, model_parameters=model_parameters['channel1']):
                
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla']
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
        
        
        def channel_2(features, model_parameters=model_parameters['channel2']):
                
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla']
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
            
        
        outputs1 = channel_1(features)
        outputs2 = channel_2(features = layers.Concatenate(axis=1)([features, outputs1]))
        
        outputs = layers.Concatenate(axis=1)([outputs1, outputs2]) 
        
        # if model_parameters['distribution_output']:
        #     outputs = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., 0], scale=1e-3 + tf.math.softplus(t[..., 1])))(outputs)
        
        ### Output Layer ###
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs1)
            
        model = self.update_model()
        
        return model        



    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['DualChannelCNNHead'] = model
        
        return model
        
    
    
class DualChannelMultiHead(UncertaintyModel):
    """
    Multi Headed Attention inspired model with dual channel body.
    - channel1 - Can be Epistemic / Deterministic - predicts mean
    - channel2 - Can be Epistemic / Deterministic - predicts stddev
    
    
    """
    
    def __init__(self, **kwargs):
    
        super(DualChannelMultiHead, self).__init__(**kwargs)
        
        self.model_parameters = {'use_current_rate' : False,
                                 'channel1' : {'dense_units_vanilla' : [10,10],
                                               'activity_regularizer_vanilla' : tf.keras.regularizers.L2(1e-5),
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [1],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None},
                                 'channel2' : {'dense_units_vanilla' : [5],
                                               'activity_regularizer_vanilla' : None,
                                               'activation_vanilla' : "relu",
                                               'dense_units_bayesian': [],
                                               'activity_regularizer_bayesian' : None,
                                               'activation_bayesian' : None}
                                 }
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 100,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0.2,
                                    'verbose' : 1}
    


    def create_core(self, model_parameters=None):
        """
        Create core model.

        Parameters
        ----------
        model_parameters : dict, optional
            Model Parameters. The default is None.

        Returns
        -------
        model : tf.Keras.Model
            creates a model and adds it to models attribute.

        """
        
        if not model_parameters:
            model_parameters = self.model_parameters
        
            
        self.inputs = layers.Input(shape=(self.tensors['features'].shape[1],
                                     self.tensors['features'].shape[2]))
        
        features = self.inputs

        class CurrentStdDev(tf.keras.layers.Layer):
            
            def __init__(self):
                super(CurrentStdDev, self).__init__()
                
            def build(self, input_shape):
                self.w_train = self.add_weight(name='current_std_dev_w', shape=(1,), initializer="ones", trainable=True)
                
            def call(self, inputs):
                return tf.math.reduce_std(inputs[..., 0], axis=1) * self.w_train
                
        features1 = CurrentStdDev()(features)
        features1 = layers.Flatten()(features1)
        
        
        if model_parameters['use_current_rate']:
            class CurrentRate(tf.keras.layers.Layer):
                
                def __init__(self):
                    super(CurrentRate, self).__init__()
                    
                def build(self, input_shape):
                    self.w_static = tf.constant([3./12., -16./12., 36./12., -48./12., 25./12.], dtype=tf.float32)
                    self.w_train = self.add_weight(name='current_rate_w', shape=(1,), initializer="ones", trainable=True)
                    
                def call(self, inputs):
                    return tf.math.reduce_sum(tf.multiply(inputs[..., 0], self.w_static), axis=1) * self.w_train
                
            features2 = CurrentRate()(features)
            features2 = layers.Flatten()(features2)
            
        else:
            class CurrentMean(tf.keras.layers.Layer):
            
                def __init__(self):
                    super(CurrentMean, self).__init__()
                
                def build(self, input_shape):
                    self.w_train = self.add_weight(name='current_mean_w', shape=(1,), initializer="ones", trainable=True)
                    
                def call(self, inputs):
                    return tf.math.reduce_mean(inputs[..., 0], axis=1) * self.w_train
                    
            features2 = CurrentMean()(features)
            features2 = layers.Flatten()(features2)
        
        
        class WeightedStates(tf.keras.layers.Layer):
            
            def __init__(self):
                super(WeightedStates, self).__init__()
            
            def build(self, input_shape):
               self.w_train = self.add_weight(name='weighted_states_w', shape=(1,3), initializer="ones", trainable=True)
               self.b_train = self.add_weight(name='weighted_states_b', shape=(1,3), initializer="zeros", trainable=True)
            
            def call(self, inputs):
                return tf.multiply(inputs[..., -1, :], self.w_train) + self.b_train
        
        features3 = WeightedStates()(features)
        features3 = layers.Flatten()(features3)
        
        features = layers.Concatenate(axis=1)([features1, features2, features3]) 
        
        
        prior = self._Independent
        posterior = self._Independent_loc_scale
        
        
        def channel_1(features, model_parameters=model_parameters['channel1']):
                
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla']
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
        
        
        def channel_2(features, model_parameters=model_parameters['channel2']):
            
            if model_parameters['dense_units_vanilla']:
                for units in model_parameters['dense_units_vanilla']:
                    
                    features = layers.Dense(
                        units = units,
                        activation = model_parameters['activation_vanilla'],
                        activity_regularizer = model_parameters['activity_regularizer_vanilla'],
                        )(features)
                
                
            if model_parameters['dense_units_bayesian']:
                for units in model_parameters['dense_units_bayesian']:
                   
                    features = tfp.layers.DenseVariational(
                        units = units,
                        activation = model_parameters['activation_bayesian'],
                        make_prior_fn = prior,
                        make_posterior_fn = posterior,
                        kl_weight = 1/(self.tensors['features'][0]),
                        activity_regularizer = model_parameters['activity_regularizer_bayesian']
                        )(features)
                
            ### Output Layer ###
            features = layers.Dense(units=self.tensors['targets'].shape[-1],
                                    activation=None)(features)
            
            return features
            
        
        outputs1 = channel_1(features)
        outputs2 = channel_2(features = layers.Concatenate(axis=1)([features, outputs1]))
        
        outputs = layers.Concatenate(axis=1)([outputs1, outputs2])
        
        ### Output Layer ###
        self.outputs_1 = tfp.layers.IndependentNormal(1)(outputs)
        
        self.outputs_2 = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-3))(outputs1)
            
        model = self.update_model()
        
        return model        



    def update_model(self, probabilstic_mode=False):
        
        model = keras.Model(inputs=self.inputs, outputs=self.outputs_2)
        model.trainable = True
        
        if probabilstic_mode:
            model.trainable = False
            model = keras.Model(inputs=self.inputs, outputs=self.outputs_1)
        
        self.models['DualChannelMultiHead'] = model
        
        return model




#--- Work in Progress ---#
class ConvAutoEncoder(UncertaintyModel):
    """
    Convolutional Autoencoder for unsupervised learning.
    
    
    """
    
    
    def __init__(self, **kwargs):
    
        super(ConvAutoEncoder, self).__init__(**kwargs)
        
        self.model_parameters = {'distribution_output' : True}
        
        
        self.training_parameters = {'KL_loss' : True,
                                    'batch_size' : 1024,
                                    'epochs' : 500,
                                    'learning_rate' : 0.01,     
                                    'validation_split' : 0,
                                    'verbose' : 1}
        
        
    def create_core(self, model_paramters=None):
        
        class AutoEncoder(tf.keras.Model):
          
            def __init__(self, input_dim, latent_dim, output_dim):
        
                super(AutoEncoder, self).__init__()
                
                self.encoder = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(input_dim[0],input_dim[1])),
                    tf.keras.layers.Conv1D(filters=input_dim[1], kernel_size=input_dim[1], activation='relu', padding='causal'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(units=latent_dim**3, activation='relu'),
                    tf.keras.layers.Dense(units=latent_dim**2, activation='relu'),
                    tf.keras.layers.Dense(units=latent_dim**1)
                    ])
                
                self.decoder_1 = tf.keras.Sequential([
                     tf.keras.layers.InputLayer(input_shape=(latent_dim**1)),
                     tf.keras.layers.Dense(units=latent_dim**2, activation='relu'),
                     tf.keras.layers.Dense(units=latent_dim**3, activation='relu'),
                     tf.keras.layers.Dense(units=self.encoder.layers[1].output_shape[1], activation='relu'),
                     tf.keras.layers.Reshape(target_shape=(input_dim[0],input_dim[1])),
                     tf.keras.layers.Conv1DTranspose(filters=input_dim[1], kernel_size=input_dim[1], activation='relu', padding='same')
                     ])
                
                self.decoder_2 = tf.keras.Sequential([
                     tf.keras.layers.InputLayer(input_shape=(latent_dim**1)),
                     tf.keras.layers.Dense(units=latent_dim**2, activation='relu'),
                     tf.keras.layers.Dense(units=latent_dim**3, activation='relu'),
                     tf.keras.layers.Dense(units=output_dim, activation='relu'),
                     tf.keras.layers.Reshape(target_shape=(output_dim, 1))
                             ])
                
                
          
            def call(self, inputs):
                    
                encoded = self.encoder(inputs)
                
                decoded_1 = self.decoder_1(encoded)
                decoded_2 = self.decoder_2(encoded)
                
                return decoded_1, decoded_2
            
            
        model = AutoEncoder(input_dim=[self.tensors['features'].shape[1],
                                       self.tensors['features'].shape[2]], 
                            latent_dim=2, 
                            output_dim=self.tensors['targets'].shape[1])
        
        self.models['ConvAutoEncoder'] = model
        
        
        
    def fit_model(self, training_parameters=None, model_name=None):
        
        if not training_parameters:
            training_parameters = self.training_parameters
        
        model = self.models[model_name]
    
        loss = tf.keras.losses.MeanSquaredError()
    
        model.compile(
                optimizer = keras.optimizers.Adam(learning_rate=training_parameters['learning_rate']),
                loss = [loss, loss],
                loss_weights = [1, 1],
                metrics = [keras.metrics.RootMeanSquaredError()]
                )
        
        print("Start training the model...")
        
        latent_dim_list = []
        
        for i in range(len(self.tensors['features'])):
            
            history = model.fit(
                x = tf.reshape(self.tensors['features'][i], [1, self.tensors['features'].shape[1], self.tensors['features'].shape[2]]),
                y = [tf.reshape(self.tensors['targets'][i], [1, self.tensors['targets'].shape[1], self.tensors['targets'].shape[2]]),
                     tf.reshape(self.tensors['features'][i], [1, self.tensors['features'].shape[1], self.tensors['features'].shape[2]])],
                epochs = 100,
                verbose = 0,
                )
            
            pc = model.encoder(tf.reshape(self.tensors['features'][i], [1, self.tensors['features'].shape[1], self.tensors['features'].shape[2]]))
            loss = history.history['loss'][-1]
            
            latent_dim_list.append([pc.numpy(), loss])
            
            print(str(i) + "/" + str(len(self.tensors['features'])) + " ... loss value : "+ str(loss))
            
        print("Model training finished.")
        
        self.models[model_name] = model
        
        return latent_dim_list
        

#%% Extras
"""



    def _prior_1(self, kernel_size, bias_size, dtype=None):
        
        n = kernel_size + bias_size 
        
        prior_model = keras.Sequential([
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n))
                    )
                ])
                
        return prior_model



    def _prior_2(self, kernel_size, bias_size=0, dtype=None):
        
        n = kernel_size + bias_size
        
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1e-3),
                reinterpreted_batch_ndims=1)),
        ]) 
    
    
    
    def _prior_3(self, kernel_size, bias_size=0, dtype=None):
        
        n = kernel_size + bias_size
        
        # return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1e-3),
        #                                  reinterpreted_batch_ndims=1)
        
        prior_model = keras.Sequential([
               tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                   tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1e-3),
                   reinterpreted_batch_ndims=1)
                   )
               ])
               
        return prior_model


    def _posterior_1(self, kernel_size, bias_size, dtype=None):
        
        n = kernel_size + bias_size
        
        posterior_model = keras.Sequential([
                tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
                tfp.layers.MultivariateNormalTriL(n),
            ])
        
        return posterior_model    

    

    def _posterior_2(self, kernel_size, bias_size=0, dtype=None):
        
        n = kernel_size + bias_size
        
        return tf.keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1e-3),
                reinterpreted_batch_ndims=1))
            ])
        
        
        # c = np.log(np.expm1(1.))
        
        # return tf.keras.Sequential([
        #     tfp.layers.VariableLayer(2 * n, dtype=dtype),
        #     tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        #         tfd.Normal(loc=t[..., :n],
        #                    scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
        #         reinterpreted_batch_ndims=1)),
        # ])
    
    
    def _posterior_3(self, kernel_size, bias_size, dtype=None):
        
        n = kernel_size + bias_size
        
        return tf.keras.Sequential([
                tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
                tfp.layers.IndependentNormal(n)
                ])



"""
