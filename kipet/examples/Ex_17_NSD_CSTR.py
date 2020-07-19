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

factor = np.random.uniform(low=0.8, high=1.2, size=len(parameters))

d_init_guess = {p.name: (p.init*factor[i], p.bounds) for i, p in enumerate(parameters)}
#d_init_guess = {p.name: (p.init, p.bounds) for i, p in enumerate(parameters)}


# NSD routine

# If there is only one parameter it does not really update - it helps if you
# let the other infomation "seep in" from the other experiments.

parameter_var_name = 'P'
options = {
    'method': 'trust-constr',
    'use_est_param': True,   # Use this to reduce model based on EP
    'use_scaling' : True,
    'cross_updating' : True,
    }

print(d_init_guess)

nsd = NSD(models, d_init_guess, parameter_var_name, options)
results, od = nsd.nested_schur_decomposition(debug=True)

print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')

#%% For paper - automatic figure and table generation

filedir = '../../../../Documents/Work/CMU Work/NSD/'
filename = 'CSTR_BR_ESC_80120r' #'CSTR_BR_ESC_fr80-120'
save_figures = True

def make_param_table(p_dict, filedir, filename):
    
    output_table = filedir + filename + '_table.tex'
    
    p_tex = {'Cfa': '$C_{A,F}$',
             'ER' : '$E/R$',
             'k' : '$k$',
             'Tfc' : '$T_{C,F}$',
             'rho' : '$\\rho$',
             'rhoc' : '$\\rho_C$',
             'Tf' : '$T_{F}$',
             'h' : '$h$',
             'delH' : '$\\Delta H$',
             }
    
    with open(output_table, 'w') as f:
    
        f.write('\\begin{table}[h!]\n')
        f.write('\\begin{center}\n')
        f.write('\\caption{The CSTR model constants}\n')
        f.write('\\label{tab:table_cstr_constants}\n')
        f.write('\\begin{tabular}{c|c}\n')
        f.write('\\toprule\n')
        f.write('\\textbf{Constant} & \\textbf{Symbol}\\\\\n')
        f.write('\\midrule\n')
        
        for p, v in p_dict.items():
        
            f.write('%s & %0.4f' % (p_tex[p], v))
            f.write('\\\\\n')
        
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{center}\n')
        f.write('\\end{table}\n')

    return None

if save_figures:

    nsd.plot_paths(filedir + filename)
    make_param_table(nsd.parameters_opt, filedir, filename)

