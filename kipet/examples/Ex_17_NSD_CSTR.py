"""
Example 18: Nested Schur Decomposition for the CSTR Model

Uses two simulated experiments to show how the NSD method can be used to
combine data collected in different methods into a single parameter fitting
problem.

"""
import numpy as np
import pandas as pd

from kipet.library.EstimationPotential import reduce_models
from kipet.library.NestedSchurDecomposition import NestedSchurDecomposition as NSD
from kipet.examples.Ex_17_CSTR_setup import make_model_dict

# Models and data are generated in make_model_dict from Ex_17_CSTR_setup
# One is a CSTR experiment with many temperature measurements and three
# concentration measuremnets. The second is an isothermal reaction with three
# concentration measurements as well.

# You could use the following models directly in NSD:
models, parameters = make_model_dict() 

# These are my fixed "randomizations" of the initial parameter values
factor = np.array([1.020269,   0.9819293,  0.96960294, 
                   0.82314446, 0.98043732, 0.83740861,
                   0.91065638, 0.98690073, 0.83397106])

# Set up the parameters in the following manner:
# {key : (initial value, (lower bound, upper bound))}
# e.g. {'k1' : (0.5, (0.01, 2))}
d_init_guess = {p.name: (p.init*factor[i], p.bounds) for i, p in enumerate(parameters)}

# If using KIPET, the models may be reduced (recommended)
# The routine in EstimationPotential can now be called using reduce_models:
#models, param_data = reduce_models(models, parameter_dict=d_init_guess) 

# Update the initial parameter values using the averages from model reduction:
#d_init_guess = param_data['initial_values']

# NSD - First declare options
# This is important - the name of the global parameter set:
parameter_var_name = 'P'

# Other options should be placed in a dict:
options = {
    'method': 'trust-constr',
    'use_scaling' : True,
    'cross_updating' : True,
    'conditioning' : False,
    'conditioning_Q': 10,
    }

# Declare NSD instance with dict of models, the dict of parameter values and bounds
# the parameter set name, and NSD options
nsd = NSD(models, d_init_guess, parameter_var_name, options)

# Call NSD method to calculate the optimal parameter values:
results, od = nsd.nested_schur_decomposition(debug=True)

# Final parameter values can also be accessed using nsd.parameters_opt attr
print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')