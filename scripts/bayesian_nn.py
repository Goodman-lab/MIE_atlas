#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Jan 2022
Modified May 2022
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

bnn_test_r2_list = []
bnn_test_mae_list = []

bnn_val_r2_list = []
bnn_val_mae_list = []

bnn_test_cali_r2_list = []
bnn_val_cali_r2_list = []
bnn_test_oracle_auc_list = []
bnn_val_oracle_auc_list = []
bnn_test_eff_list = []
bnn_val_eff_list = []
bnn_test_disp_list = []
bnn_val_disp_list = []
bnn_test_gmp_list = []
bnn_val_gmp_list = []

bnn_test_alea_mean_list = []
bnn_test_alea_std_list = []
bnn_test_aspars_r_list = []
bnn_test_epis_mean_list = []
bnn_test_epis_std_list = []
bnn_test_espars_r_list = []

bnn_val_alea_mean_list = []
bnn_val_alea_std_list = []
bnn_val_aspars_r_list = []
bnn_val_epis_mean_list = []
bnn_val_epis_std_list = []
bnn_val_espars_r_list = []


print("Beginning model building...")
print("There are", len(target_list), "targets to build for.")


    
with tf.device('/CPU:0'):

    for target in target_list: 

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
        tsvd_on = True

        
        if tsvd_on: 
            d_metric = 'euclidean'
            svd = TruncatedSVD(n_components=512)
            svd.fit(X)
            X = svd.transform(X)
            X_val = svd.transform(X_val)
            data_spars_neighbors = X.shape[0]//16

        else: 
            d_metric = 'jaccard'
            data_spars_neighbors = X.shape[0]//16

        
        
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
        #bnn.summary()
        
        
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
        

        # Get test metrics
        
        y_mean_test, y_sigma_test = bnn.predict_normal_distr(X_test, n_samples=1000)
        bnn_test_mae = mean_absolute_error(y_test, y_mean_test)
        bnn_test_r2 = r2_score(y_test, y_mean_test)
        
        bnn_test_cali_r2, bnn_test_eff, bnn_test_disp, bnn_test_gmp = all_uncertainty_metrics(bnn, X_test, y_test)
        
        bnn_test_oracle_auc = oracle_gmp_auc(bnn, X_test, y_test)
        
        bnn_test_alea, bnn_test_alea_mean, bnn_test_alea_std = bnn.get_aleatoric(X_test)
        bnn_test_aspars_r = alea_data_sparsity_pearsonr(bnn, X_train, y_train, X_test, n_neighbors = data_spars_neighbors, distance = d_metric)
       
        bnn_test_epis, bnn_test_epis_mean, bnn_test_epis_std = bnn.get_epistemic(X_test)
        bnn_test_espars_r = epis_data_sparsity_pearsonr(bnn, X_train, y_train, X_test, n_neighbors = data_spars_neighbors, distance = d_metric)
        

        
        bnn_test_mae_list.append(bnn_test_mae)
        bnn_test_r2_list.append(bnn_test_r2)
        bnn_test_cali_r2_list.append(bnn_test_cali_r2)
        bnn_test_eff_list.append(bnn_test_eff)
        bnn_test_disp_list.append(bnn_test_disp)
        bnn_test_gmp_list.append(bnn_test_gmp)
        bnn_test_oracle_auc_list.append(bnn_test_oracle_auc)
        bnn_test_alea_mean_list.append(bnn_test_alea_mean)
        bnn_test_alea_std_list.append(bnn_test_alea_std)
        bnn_test_aspars_r_list.append(bnn_test_aspars_r)
        bnn_test_epis_mean_list.append(bnn_test_epis_mean)
        bnn_test_epis_std_list.append(bnn_test_epis_std)
        bnn_test_espars_r_list.append(bnn_test_espars_r)
                
        
        #print("test cali curve")
        #cali_curve(bnn, X_test, y_test)
        
        # Get val metrics
        
        y_mean_val, y_sigma_val = bnn.predict_normal_distr(X_val, n_samples=1000)
        bnn_val_mae = mean_absolute_error(y_val, y_mean_val)
        bnn_val_r2 = r2_score(y_val, y_mean_val)
        
        bnn_val_cali_r2, bnn_val_eff, bnn_val_disp, bnn_val_gmp = all_uncertainty_metrics(bnn, X_val, y_val)
        
        bnn_val_oracle_auc = oracle_gmp_auc(bnn, X_val, y_val)
        
        bnn_val_alea, bnn_val_alea_mean, bnn_val_alea_std = bnn.get_aleatoric(X_val)
        bnn_val_aspars_r = alea_data_sparsity_pearsonr(bnn, X_train, y_train, X_val, n_neighbors = data_spars_neighbors, distance = d_metric)
        
        bnn_val_epis, bnn_val_epis_mean, bnn_val_epis_std = bnn.get_epistemic(X_val)
        bnn_val_espars_r = epis_data_sparsity_pearsonr(bnn, X_train, y_train, X_val, n_neighbors = data_spars_neighbors, distance = d_metric)
        
        bnn_val_mae_list.append(bnn_val_mae)
        bnn_val_r2_list.append(bnn_val_r2)
        bnn_val_cali_r2_list.append(bnn_val_cali_r2)
        bnn_val_eff_list.append(bnn_val_eff)
        bnn_val_disp_list.append(bnn_val_disp)
        bnn_val_gmp_list.append(bnn_val_gmp)
        bnn_val_oracle_auc_list.append(bnn_val_oracle_auc)
        bnn_val_alea_mean_list.append(bnn_val_alea_mean)
        bnn_val_alea_std_list.append(bnn_val_alea_std)
        bnn_val_aspars_r_list.append(bnn_val_aspars_r)
        bnn_val_epis_mean_list.append(bnn_val_epis_mean)
        bnn_val_epis_std_list.append(bnn_val_epis_std)
        bnn_val_espars_r_list.append(bnn_val_espars_r)
        
        
        print("val cali curve")
        cali_curve(bnn, X_val, y_val)
        print("val data sparsity plot")
        epis_data_sparsity_plot(bnn, X_train, y_train, X_val, n_neighbors = data_spars_neighbors, distance = d_metric)
        
        # Reset session
        tf.keras.backend.clear_session()

# Print results to sheet
metrics_list = ["BNN test MAE", "BNN test R2", "BNN test cali R2", "BNN test oracle error AUC", "BNN test efficiency", "BNN test dispersion", "BNN test gmean probs", "BNN test aleatoric mean", "BNN test aleatoric std", "BNN test alea spars r", "BNN test epistemic mean", "BNN test epistemic std", "BNN test epis sparsity r", 
                "BNN val MAE", "BNN val R2", "BNN val cali R2", "BNN val efficiency", "BNN val dispersion", "BNN val gmean probs", "BNN val oracle error AUC", "BNN val aleatoric mean", "BNN val aleatoric std", "BNN val alea spars r", "BNN val epistemic mean", "BNN val epistemic std", "BNN val epis sparsity r"]

print("END")

results_df = pd.DataFrame(np.column_stack([bnn_test_mae_list, bnn_test_r2_list, bnn_test_cali_r2_list, bnn_test_oracle_auc_list, bnn_test_eff_list, bnn_test_disp_list, bnn_test_gmp_list, bnn_test_alea_mean_list, bnn_test_alea_std_list, bnn_test_aspars_r_list, bnn_test_epis_mean_list, bnn_test_epis_std_list, bnn_test_espars_r_list,
                                           bnn_val_mae_list, bnn_val_r2_list, bnn_val_cali_r2_list, bnn_val_eff_list, bnn_val_disp_list, bnn_val_gmp_list, bnn_val_oracle_auc_list, bnn_val_alea_mean_list, bnn_val_alea_std_list, bnn_val_aspars_r_list, bnn_val_epis_mean_list, bnn_val_epis_std_list, bnn_val_espars_r_list]).T, 
                          index = metrics_list, 
                          columns=target_list)

results_mean = results_df.mean(axis = 1)
results_std = results_df.std(axis=1)

results_df['MEAN'] = results_mean
results_df['STD'] = results_std
print(results_df)

from datetime import date
today = date.today().strftime("%d_%m_%Y")
results_df.to_csv("/home/cg588/MIE_atlas/tests/bayesian_nn/"+today+".csv")


'''

# Inspect predictions
y_mean_val, y_sigma_val = bnn.predict_normal_distr(X_val, n_samples = 500)
y_mean_train, y_sigma_train = bnn.predict_normal_distr(X_train, n_samples = 500)
y_mean_test, y_sigma_test = bnn.predict_normal_distr(X_test, n_samples = 500)

# Plot experimental vs predicted values for validation data
plt.figure()
plt.scatter(y_val,y_mean_val, marker='o')
plt.plot([3, 10], [3, 10], c = 'k', lw = 2)
plt.show()


print("--------------------")
print("Training Set MAE:")
print(mean_absolute_error(y_train, y_mean_train))
print("Test Set MAE:")
print(mean_absolute_error(y_test, y_mean_test))
print("Validation Set MAE:")
print(mean_absolute_error(y_val, y_mean_val))


# End the cycle
tf.keras.backend.clear_session()

# Endgame
print("END")

'''


#%%

from MIE_atlas.tests.uncertainty_metrics import *

with tf.device('/CPU:0'): 
    y_mu, y_sigma = bnn.predict_normal_distr(X_val[0].reshape(1,-1), n_samples = 500)
    cdfs = [st.norm.cdf(5, y_mu[i], y_sigma[i]) for i in range(1)]
#%%
print(y_mu + y_sigma)