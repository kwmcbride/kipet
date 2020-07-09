"""
Example 18: Nested Schur Decomposition for the CSTR Model

Uses three simulated experiments to show how the NSD method can be used to
combine data collected in different methods into a single parameter fitting
problem.

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

# If there is only one parameter it does not really update - it helps if you
# let the other infomation "seep in" from the other experiments.

parameter_var_name = 'P'
options = {
    'method': 'trust-constr',
    'use_est_param': True,      # Use this to reduce model based on EP
    }

nsd = NSD(models, d_init_guess, parameter_var_name, options)
results, od = nsd.nested_schur_decomposition(debug=True)

print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')

nsd.plot_paths()