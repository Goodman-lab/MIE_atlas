#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:26:41 2022

Bayesian NN class

@author: cg588
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tqdm
import scipy.stats as st




class BayesianNN(tf.keras.Sequential): 
    
    # Bayesian neural network: 
    # Feed forward network where weights and biases are distributions and 
    # parameters of these distributions are learnt while training
    # 
    # Made up of series of DenseVariational layers
    # Input: features as a (1, n) vector
    # Output: posterior distribution
    # Sample many outputs for prediction. 
    
    

    def predict_normal_distr(self, X, n_samples = 1000): 
        # Get prediction for X in the form of (mean, stdev)
        #
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # return y_mu: mean of the sampled posterior
        # return y_sigma: standard deviation of the sampled posterior
        
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i]= np.squeeze(super().predict(X))
            
        y_mu = np.array([np.mean(r) for r in samples])
        y_sigma = np.array([np.std(r) for r in samples])
        
        return y_mu, y_sigma
    
    def predict_central_interval(self, X, n_samples = 1000, alpha = 0.05): 
        # Get confidence intervals for molecules in X
        #
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # parameter alpha: 1-alpha is the confidence of this interval
        # return intervals: list of [lower, upper] intervals for each molecule in X
        
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i]= np.squeeze(super().predict(X))
            
        intervals = []
        for sample in samples: 
            interval = st.norm.interval(1-alpha, loc = np.mean(sample), scale = np.std(sample))
            intervals.append(interval)
        
        return np.array(intervals)
    
    def get_aleatoric(self, X, n_samples = 1000): 
        # Get aleatoric uncertainties for each molecule in X, 
        # as well as mean and stdev for all molecules in X
        # Calculated from the mean of variances from each sample. 
        #
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # return alea: list of sampled aleatoric uncertainties for molecules in X
        # return alea_mu: mean of all aleatoric uncertainties for molecules in X
        # return alea_sigma: standard deviation of all aleatoric uncertainties for molecules in X
        
        
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i]= np.squeeze(self(X).stddev())
                    
        alea = np.array([np.mean(r) for r in samples])
        alea_mu = np.mean(alea)
        alea_sigma = np.std(alea)
        
        return alea, alea_mu, alea_sigma

    def get_epistemic(self, X, n_samples = 1000): 
        # Gets epistemic uncertainties for each molecule in X, 
        # as well as mean and stdev for all molecules in X
        # Calculated from the variance of means from each sample. 
        # 
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # return epis: list of sampled epistemic uncertainties for molecules in X
        # return epis_mu: mean of all epistemic uncertainties for molecules in X
        # return epis_sigma: standard deviation of all epistemic uncertainties for molecules in X
    
        
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i]= np.squeeze(self(X).mean())
                    
        epis = np.array([np.std(r) for r in samples])
        epis_mu = np.mean(epis)
        epis_sigma = np.std(epis)
        
        return epis, epis_mu, epis_sigma
    
    def calibrate_epistemic(self, X, n_samples = 1000): 
        # Store epistemic uncertainties for molecules in training data
        # in order to get percentile for new examples
        # 
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # return self.epis_mu: stored mean of all epistemic uncertainties for training data
        # return self.epis_sigma: stored standard deviation of all epistemic uncertainties for training data
        
        self.training_epis = []
        self.training_epis, self.epis_mu, self.epis_sigma = self.get_epistemic(X, n_samples = n_samples)
        self.training_epis.sort()
        return self.epis_mu, self.epis_sigma
    
    def get_epistemic_percentile(self, X, n_samples = 1000): 
        # Read stored epistemic uncertainties to get epistemic uncertainty 
        # percentile for new examples
        # 
        # input X: features of molecules of interest
        # parameter n_samples: number of times to sample the network
        # return percentiles: list of percentiles of epistemic uncertainties for molecules in X
        
        
        X_alea, mu, sigma = self.get_epistemic(X, n_samples = n_samples)
        percentiles = st.percentileofscore(self.training_epis, X_alea, kind = 'mean')
        
        return percentiles

    def predict_cdf(self, X, threshold, n_samples = 1000): 
        # Estimate the probability of exceeding critical threshold P(y < threshold) by modelling y as N(y_mu, y_sigma)
        # To get P(y > threshold) use 1-cdf 
        #
        # input X: features of molecules of interest
        # parameter threshold: threshold of comparison
        # parameter n_samples: number of times to sample the network
        # return cdfs: list of cdf for molecules in X
        
        samples = np.zeros((len(X), n_samples))
        for i in range(n_samples): 
            samples[:,i]= np.squeeze(super().predict(X))
            
        y_mu = np.array([np.mean(r) for r in samples])
        y_sigma = np.array([np.std(r) for r in samples])
        
        
        cdfs = [st.norm.cdf(threshold, y_mu[i], y_sigma[i]) for i in range(len(X))]
        
        return cdfs
                

    # EXTRA EXPERIMENTAL FUNCTIONS / UNFINISHED CODE
    
    def _get_latent(self, X, n_samples = 1000): 
        # EXPERIMENTAL: use latent space coordinates as distance measure
        # doesn't actually work very well
        
        samples = np.zeros((len(X), self.get_layer(index=-3).units, n_samples))
        last_layer_out = tf.keras.Model(self.input, 
                            self.get_layer(index=-3).output)
        
        for i in range(n_samples): 
            samples[:,:,i] = last_layer_out(X).numpy()
            
        latent = np.mean(samples, axis = 2)    
        
        return latent
    

    
    def _all_predictions(self, X, threshold, n_samples = 1000): 
        # get predictions: mean, std, aleatoric, epistemic, cdf
        
        return
        