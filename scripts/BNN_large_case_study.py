

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022
@author: Charles Gong

Based on Tim's BNN
Created Oct 2020
@author: Timothy E H Allen / Alistair M Middleton
"""
#%%

# Import modules

import sys
try: 
    sys.path.index('/home/cg588')
except ValueError:
    sys.path.append('/home/cg588')


from MIE_atlas.featurizers.molfeaturizer import MorganFPFeaturizer
from MIE_atlas.tests.uncertainty_metrics import *
from MIE_atlas.models.BayesianNN import BayesianNN
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import tqdm
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import TruncatedSVD
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
import matplotlib.pyplot as plt

tf.enable_v2_behavior()
tfd = tfp.distributions



# HYPERPARAMETERS
rng_1 = 1989
rng_2 = 2020
validation_proportion = 0.25
neurons = 10
hidden_layers = 3
LR = 0.005
epochs = 200
batch_size= 100


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])), 
            reinterpreted_batch_ndims=1)),
    ])

# Specify a non-trainable prior

def prior_not_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    pi = .5
    return tf.keras.Sequential([
        tfp.layers.DistributionLambda(lambda t: tfd.Mixture(
            cat=tfd.Categorical(probs=[pi, 1. - pi]),
            components=[tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=.001 * tf.ones(n)),
                        tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=1.5 * tf.ones(n))
                        ])
                                      )
    ])


def negloglik(y, rv_y):
    return -rv_y.log_prob(y)



# Specify model architecture
def build_model(hidden_layers): 
    if hidden_layers == 1:
        model_aleatoric_epistemic = BayesianNN([
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense1'),
            tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, name = 'mve'),
            tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-5 + tf.math.softplus(1e-1 * t[..., 1:])), name = 'posterior') 
            ])
        return model_aleatoric_epistemic
    
    elif hidden_layers == 2:
        model_aleatoric_epistemic = BayesianNN([
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense1'),
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense2'),
            tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, name = 'mve'),
            tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-5 + tf.math.softplus(1e-1 * t[..., 1:])), name = 'posterior') 
            ])
        return model_aleatoric_epistemic
        
    elif hidden_layers == 3:
        model_aleatoric_epistemic = BayesianNN([
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense1'),
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense2'),
            tfp.layers.DenseVariational(neurons, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, activation='relu', name = 'dense3'),
            tfp.layers.DenseVariational(1 + 1, posterior_mean_field, prior_not_trainable, kl_weight=kl_loss_weight, name = 'mve'),
            tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-5 + tf.math.softplus(1e-1 * t[..., 1:])), name = 'posterior') 
            ])
        return model_aleatoric_epistemic
        
    else:
        print("Number of hidden layers outside this model scope, please choose 1, 2, or 3")
        return None

# Learning rate scheduler
def scheduler(epoch, lr):
  if epoch < 50:
    return lr
  else:
    return lr * tf.math.exp(-0.01)


# Define model scope

target_list = ['AChE', 'ADORA2A', 'ADRB1', 'ADRB2', 'AR', 'CHRM1', 'CHRM2', 'CHRM3', 'DD1R', 'DD2R', 'EDNRA', 'HRH1', 'HTR2A', 'KCNH2', 'LCK', 'NR3C1', 'OPRD1', 'OPRM1', 'SLC6A2', 'SLC6A3', 'SLC6A4']
#target_list = ['AChE']
case_study_path = '/home/cg588/MIE_atlas/tests/case_study/Toolbox_chemicals/'


with open(case_study_path+'Toolbox_chemicals.smi') as f:
    case_study_chems = [x.strip('\n') for x in f.readlines()] # get list of smiles to evaluate


print("Beginning model building...")
print("There are", len(target_list), "targets to build for.")


for target in target_list: 
    with tf.device('/CPU:0'):
    
        # Reset previous session
        tf.keras.backend.clear_session()
    
        # Get data
        
        print("Building uncertainty for target:", target)
        raw_data = pd.read_csv("/home/cg588/MIE_atlas/data/input_data/"+target+".csv")
        val_data = pd.read_csv("/home/cg588/MIE_atlas/data/ext_validation/"+target+".csv")
        model_path = "/home/cg588/MIE_atlas/models/saved_models" + target + "_bnn"
        
        featurizer = MorganFPFeaturizer(radius = 2, nBits = 8000)
        X = featurizer.transform(raw_data['SMILES'])
        y = raw_data['P(Act)'].values
        X_val = featurizer.transform(val_data['SMILES'])
        y_val = val_data['P(Act)'].values
        
        assert y.shape[0] == X.shape[0]
        assert y_val.shape[0] == X_val.shape[0]
        
        X, y = shuffle(X, y, random_state=rng_1)
        
        
        # Optional SVD
        svd = TruncatedSVD(n_components=512)
        svd.fit(X)
        X = svd.transform(X)
        X_val = svd.transform(X_val)
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rng_2)
        
        
        # Note that the kl_loss_weight value here is 1 over the size of the entire dataset, not just the batch size.
        
        kl_loss_weight = 1 / X_train.shape[0]
        
        # Train model
        bnn = build_model(hidden_layers)
        bnn.compile(optimizer=tf.optimizers.Adam(learning_rate=LR), loss=negloglik, metrics=['mse'])
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                           patience = 1, 
                                                           min_delta = 10, 
                                                           factor = tf.math.exp(-0.01))
        lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', 
                                                           patience = 1, 
                                                           min_delta = 1, 
                                                           factor = tf.math.exp(-0.02))
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                      patience = 30, 
                                                      min_delta = 1, restore_best_weights = True)
        history = bnn.fit(X_train, y_train, epochs=epochs, 
                          callbacks = [lr_schedule, lr_on_plateau, early_stop], 
                          verbose=False, validation_data=(X_test, y_test))
        bnn.summary()
        
        
        # NEED TO FIX SAVE MODEL, IT'S BROKEN
        #model.save(model_path, save_format = "tf")
        
        # Plot history of loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
        
        # Plot history of MSE values
        plt.plot(history.history['mse'])
        plt.plot(history.history['val_mse'])
        plt.title('model MSE')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
    
        #print("val cali curve")
        #cali_curve(bnn, X_val, y_val)
        print("val data sparsity plot")
        epis_data_sparsity_plot(bnn, X_train, y_train, X_val, n_neighbors = X_train.shape[0]//20, distance = 'euclidean')
        
        bnn.calibrate_epistemic(X_train, n_samples = 500)

        means = []
        stddevs = []
        cis_left = []
        cis_right = []
        epis_percentiles = []
        p_riskys = []
        for interesting in case_study_chems: 
            X_int = featurizer.transform([interesting])
            X_int = svd.transform(X_int).reshape(1,-1)
            
            y_mu, y_sigma = bnn.predict_normal_distr(X_int, n_samples = 1000)
            epis_perc = bnn.get_epistemic_percentile(X_int, n_samples = 1000)
            y_prob = np.array([1]) - bnn.predict_cdf(X_int, 5, n_samples = 1000)

            means.append(np.around(y_mu[0], 3))
            stddevs.append(np.around(y_sigma[0], 3))
            cis_left.append(np.around(y_mu[0] - 2*y_sigma[0], 3))
            cis_right.append(np.around(y_mu[0] + 2*y_sigma[0], 3))
            epis_percentiles.append(np.around(epis_perc, 1))
            p_riskys.append(np.around(y_prob[0], 3))
        
        result_df = pd.DataFrame({
            'Molecule': case_study_chems, 
            'Predicted Mean': means, 
            'Predicted Std Dev': stddevs, 
            '95 CI Lower': cis_left, 
            '95 CI Upper': cis_right, 
            'Epistemic Percentile': epis_percentiles, 
            'P_risky at threshold of 5': p_riskys})
 
        result_df.to_csv(case_study_path + target + '.csv', index = False)
        
#%%

