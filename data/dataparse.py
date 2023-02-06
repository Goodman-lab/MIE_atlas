#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:59:51 2021

Merge datasets from teha2/chemical_toxicology/Bayesian%20Learning

@author: cg588
"""

import pandas as pd
import numpy as np

target_list = ['AChE', 'ADORA2A', 'ADRB1', 'ADRB2', 'AR', 'CHRM1', 'CHRM2', 'CHRM3', 'DD1R', 'DD2R', 'EDNRA', 'HRH1', 'HTR2A', 'KCNH2', 'LCK', 'NR3C1', 'OPRD1', 'OPRM1', 'SLC6A2', 'SLC6A3', 'SLC6A4']

for target in target_list: 
    print("Merging dataset for", target)
    test_data = pd.read_csv("/home/cg588/chemical_toxicology/Bayesian Learning/Test_Sets/"+target+".csv")
    train_data = pd.read_csv("/home/cg588/chemical_toxicology/Bayesian Learning/Training_Sets/"+target+".csv")
    validation_data = pd.read_csv("/home/cg588/chemical_toxicology/Bayesian Learning/Validation_Sets/"+target+".csv")
    
    target_data = pd.concat([test_data, train_data, validation_data], ignore_index=True)
    
    target_data.to_csv("/home/cg588/MIE_atlas/data/input_data/"+target+".csv", index=False)
    

#%%

# TEST FEATURIZER

import pandas as pd
from MIE_atlas.featurizers.molfeaturizer import MorganFPFeaturizer

featurizer = MorganFPFeaturizer(radius = 2, nBits = 8000)

AChE_data = pd.read_csv("/home/cg588/MIE_atlas/data/input_data/"+"AChE"+".csv")
AChE_fp = featurizer.transform(AChE_data['SMILES'])

print(AChE_fp)
