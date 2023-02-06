#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# imports
# requires numpy, RDKit

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# MolFeaturizers

class MolFeaturizer: 
    '''
    Parent class for MolFeaturizer objects
    '''
    def __init__(self, *args, **kwargs): 
        pass
    


class MorganFPFeaturizer(MolFeaturizer):
    '''
    Morgan FP Featurizer from RDKit. 
    Input list of SMILES, returns n_mols by nBits array containing Morgan fingerprints. 
    '''
    def __init__(self, radius, nBits): 
        self.radius = radius
        self.nBits = nBits
        self.name = "Morgan FP Featurizer"
    
    def transform(self, smiles): 
        self.fp_list = []
        for smile in smiles: 
            self.mol = Chem.MolFromSmiles(smile)
            self.fp = list(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol = self.mol, radius=self.radius, nBits=self.nBits).ToBitString())
            self.fp_list.append(self.fp)
        return np.array(self.fp_list).astype(int)
        