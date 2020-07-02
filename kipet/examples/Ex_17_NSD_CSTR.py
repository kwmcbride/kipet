#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:57:47 2020

@author: kevin
"""
import numpy as np
import pandas as pd

from kipet.library.NestedSchurDecomposition import NestedSchurDecomposition as NSD
from kipet.examples.Ex_17_CSTR_setup import make_model_dict

# For simplicity, all of the models and simulated data are generated in
# the following function

models, parameters = make_model_dict() 

# This is still needed and should include the union of all parameters in all
# models
d_init_guess = {p.name: (p.init, p.bounds) for p in parameters}

# NSD routine

parameter_var_name = 'P'
options = {
    'method': 'trust-constr',
    'use_est_param': True,      # Use this to reduce model based on EP
    }

nsd = NSD(models, d_init_guess, parameter_var_name, options)
results = nsd.nested_schur_decomposition()

print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')
