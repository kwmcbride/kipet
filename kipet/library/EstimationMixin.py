"""This is a mixin for the NSD module to use the EstimationPotential module
embedded in Kipet

"""
import numpy as np
import pandas as pd

from kipet.library.EstimationPotential import EstimationPotential as EP
from kipet.library.VisitorMixins import ReplacementVisitor

global parameter_names

class EstimationMixin():
    
    def reduce_models(self):
        """Uses the EstimationPotential module to find out which parameters
        can be estimated using each experiment and reduces the model
        accordingly
        
        """
        global parameter_names
        parameters = self.d_info 
        
        all_param = set()
        all_param.update(p for p in parameters.keys())
        
        options = {
                'verbose' : True,
                        }
        
        # Loop through to perform EP on each model
        params_est = {}
        set_of_est_params = set()
        for name, model in self.models_dict.items():
            print(f'Starting EP analysis of {name}')
            
            est_param = EP(model, simulation_data=None, options=options)
            params_est[name] = est_param.estimate()
        
        # Add model's estimable parameters to global set
        for param_set in params_est.values():
            set_of_est_params.update(param_set)
            
        # Remove the non-estimable parameters from the odes
        for key, model in self.models_dict.items():
    
            for param in all_param.difference(set(params_est[key])):    
                parameter_to_change = param
                if parameter_to_change in model.P.keys():
                    change_value = [v[0] for p, v in parameters.items() if p == parameter_to_change][0]
                
                    for k, v in model.odes.items(): 
                        ep_updated_expr = self._update_expression(v.body, model.P[parameter_to_change], change_value)
                        model.odes[k] = ep_updated_expr == 0
            
                    model.parameter_names.remove(param)
                    del model.P[param]
        
        # Calculate initial values based on averages of EP output
        initial_values = pd.DataFrame(np.zeros((len(self.models_dict), len(set_of_est_params))), index=self.models_dict.keys(), columns=list(set_of_est_params))
    
        for exp, param_data in params_est.items(): 
            for param in param_data:
                initial_values.loc[exp, param] = param_data[param]
        
        dividers = dict(zip(initial_values.columns, np.count_nonzero(initial_values, axis=0)))
        
        init_val_sum = initial_values.sum()
        
        for param in dividers.keys():
            init_val_sum.loc[param] = init_val_sum.loc[param]/dividers[param]
        
        init_vals = init_val_sum.to_dict()
        init_bounds = {p: parameters[p][1] for p in parameters.keys() if p in set_of_est_params}
        
        # Redeclare the d_init_guess values using the new values provided by EP
        d_init_guess = {p: (init_vals[p], init_bounds[p]) for p in init_bounds.keys()}
        
        self.d_info = d_init_guess
        self.d_init = {k: v[0] for k, v in d_init_guess.items()}
        
        # The parameter names need to be updated as well
        parameter_names = list(self.d_init.keys())
                
        return parameter_names
    
    @staticmethod
    def _update_expression(expr, replacement_param, change_value):
        """Takes the non-estiambale parameter and replaces it with its intitial
        value
        
        Args:
            expr (pyomo constraint expr): the target ode constraint
            
            replacement_param (str): the non-estimable parameter to replace
            
            change_value (float): initial value for the above parameter
            
        Returns:
            new_expr (pyomo constraint expr): updated constraints with the
                desired parameter replaced with a float
        
        """
        visitor = ReplacementVisitor()
        visitor.change_replacement(change_value)
        visitor.change_suspect(id(replacement_param))
        new_expr = visitor.dfs_postorder_stack(expr)
         
        return new_expr 