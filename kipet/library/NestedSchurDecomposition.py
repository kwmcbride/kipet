"""
This module contains the nested Schur method for decomposing multiple
experiments into a bi-level optimization problem.

This version is adapted from the module used within Kipet - this version works
with all pyomo models and does not require Kipet.

If you want to use the EstimationPotential feature, you need kipet, but it is
not necessary to run full models.

Author: Kevin McBride 2020

# This is the standalone version...how to integrate into Kipet as a mixin?

"""
import copy
import psutil
from string import Template

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathos
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.core.base.PyomoModel import ConcreteModel

from pyomo.environ import (
    Constraint, 
    Objective,
    Param, 
    Set,
    SolverFactory,
    Suffix,
    )
from pyomo.environ import *
from pyomo.core.expr import current as EXPR

from scipy.optimize import (
    Bounds,
    minimize,
    )

global opt_dict
opt_dict = dict()
global opt_count
opt_count = -1
global param_dict
param_dict = dict()


global global_param_name
global global_constraint_name
global parameter_var_name
global global_set_name
global parameter_names
global pe_sets
global cross_updating
global Q_value
global use_conditioning
global use_mp

# If for some reason your model has these attributes, you will have a problem
global_set_name = 'global_parameter_set'
global_param_name = 'd_params_nsd_globals'
global_constraint_name = 'fix_params_to_global_nsd_constraint'

# Header template
iteration_spacer = Template('\n' + '#'*30 + ' $iter ' + '#'*30 + '\n')

# Set up multiprocessing
mp = pathos.helpers.mp
num_cpus = psutil.cpu_count(logical=False)

class NestedSchurDecomposition(object):
    
    """Nested Schur Decomposition approach to parameter fitting using multiple
    experiments
    
    """
    def __init__(self, models, d_info, param_var_name, kwargs=None):
        
        """Args:
            models (dict): A dict of pyomo Concrete models
            
            d_info: A dict of the global parameters including the iniital
                value and a tuple of the bounds, i.e. {'p1' : 2.0, (0.0, 4.0)}
        
            kwargs (dict): Optional arguments for the algorithm (incomplete)
        
        """
        # The models should be entered in as a dict (for now)
        self._orig_models = copy.deepcopy(models)
        self.models_dict = copy.deepcopy(models)
        
        # The global parameter information is needed, especially the bounds
        self.d_info = copy.copy(d_info)
        self.d_init = {k: v[0] for k, v in d_info.items()}
        
        # Arrange the kwargs
        self._kwargs = {} if kwargs is None else copy.copy(kwargs)
        
        # Options - inner problem optimization
        self.ncp = self._kwargs.pop('ncp', 3)
        self.nfe = self._kwargs.pop('nfe', 50)
        self.verbose = self._kwargs.pop('verbose', False)
        self.sens = self._kwargs.pop('use_k_aug', True)
        self.objective_name = self._kwargs.pop('objective_name', None)
        self.gtol = self._kwargs.pop('gtol', 1e-12)
        self.method = self._kwargs.pop('method', 'trust-constr')
        self.use_estimability = self._kwargs.pop('use_est_param', False)
        self.use_scaling = self._kwargs.pop('use_scaling', True)
        self.cross_updating = self._kwargs.pop('cross_updating', True)
        self.conditioning = self._kwargs.pop('conditioning', False)
        self.conditioning_Q = self._kwargs.pop('conditioning_Q', 50)
        self.use_mp = self._kwargs.pop('use_mp', False)
        
        global parameter_var_name
        parameter_var_name = param_var_name
        
        global parameter_names
        parameter_names = list(self.d_init.keys())
        
        global cross_updating
        cross_updating = self.cross_updating
        
        global pe_sets
        pe_sets = {k: parameter_names for k in self.models_dict.keys()}
        self.pe_sets = pe_sets
    
        # Run assertions that the model is correctly structured
        self._test_models()
        
        self.d_init_unscaled = self.d_init
        if self.use_scaling:
            self._scale_models()
        
        # Add the global constraints to the model
        self._add_global_constraints()
        self._prep_models()

        global use_conditioning
        use_conditioning = False
        # Q
        if self.conditioning:
            self._add_condition_terms()
            use_conditioning = True

        global Q_value
        Q_value = self.conditioning_Q

        global opt_dict
        opt_dict = dict()
        global param_dict
        param_dict = dict()
        
        global use_mp
        use_mp = self.use_mp

    def _test_models(self):
        """Sanity check on the input models"""
        
        for model in self.models_dict.values():
            # Check if the models are even models
            assert(isinstance(model, ConcreteModel) == True)
           
    def _add_global_constraints(self):
        """This adds the dummy constraints to the model forcing the local
        parameters to equal the current global parameter values
        
        """
        global global_param_name
        global parameter_var_name
        global global_constraint_name
        global global_set_name
        
        for model in self.models_dict.values():
            param_dict = {}
            for param in self.d_info.keys():
                if param in getattr(model, parameter_var_name):
                    param_dict.update({param: self.d_info[param][0]})

            setattr(model, global_set_name, Set(initialize=param_dict.keys()))

            setattr(model,
                    global_param_name, 
                    Param(getattr(model, global_set_name),
                                  initialize=param_dict,
                                  mutable=True,
                                  ))
            
            def rule_fix_global_parameters(m, k):
                
                return getattr(m, parameter_var_name)[k] - getattr(m, global_param_name)[k] == 0
                
            setattr(model, global_constraint_name, 
            Constraint(getattr(model, global_set_name), rule=rule_fix_global_parameters))
        
    def _add_condition_terms(self):
        
        global global_param_name
        
        for model in self.models_dict.values():
            for key, param in model.P.items():
            
                Q_term = self.conditioning_Q*(model.P[key] - getattr(model, global_param_name)[key])**2
                model.objective.expr += Q_term
                
        return None
        
    def _prep_models(self):
        """Prepare the model suffixes for NSD algorithm.
        
        """
        for model in self.models_dict.values():        
            model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
            model.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
            model.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
            model.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        
        return None
            
    def _generate_bounds_object(self):
        """Creates the Bounds object needed by SciPy for minimization
        
        Returns:
            bounds (scipy Bounds object): returns the parameter bounds for the
                trust-region method
        
        """
        lower_bounds = []
        upper_bounds = []
        
        for k, v in self.d_info.items():
            lower_bounds.append(v[1][0])
            upper_bounds.append(v[1][1])
        
        bounds = Bounds(lower_bounds, upper_bounds, True)
        return bounds
    
    def _scale_models(self):
        """Scale the model parameters
        
        """
        models = self.models_dict
        
        def scale_parameters(models):
            """If scaling, this multiplies the constants in model.K to each
            parameter in model.P.
            
            I am not sure if this is necessary and will look into its importance.
            """
            for k, model in models.items():
            
                #if model.K is not None:
                scale = {}
                for i in model.P:
                    scale[id(model.P[i])] = self.d_init[i]
            
                for i in model.Z:
                    scale[id(model.Z[i])] = 1
                    
                for i in model.dZdt:
                    scale[id(model.dZdt[i])] = 1
                    
                for i in model.X:
                    scale[id(model.X[i])] = 1
            
                for i in model.dXdt:
                    scale[id(model.dXdt[i])] = 1
            
                for k, v in model.odes.items(): 
                    scaled_expr = scale_expression(v.body, scale)
                    model.odes[k] = scaled_expr == 0
                
            return models
        def scale_expression(expr, scale):
            
            visitor = ScalingVisitor(scale)
            return visitor.dfs_postorder_stack(expr)
        
        self.models_dict = scale_parameters(models)

        for key, model in self.models_dict.items():
            rho = 10
            for k, v in model.P.items():
                ub = 1.5
                lb = 0.5
                model.P[k].setlb(lb)
                model.P[k].setub(ub)
                model.P[k].unfix()
                model.P[k].set_value(1)
                
                # print(v.value)
                # print(f'LB: {self.d_info[k][1][0]/self.d_init_unscaled[k]}')
                # print(f'UB: {self.d_info[k][1][1]/self.d_init_unscaled[k]}')
            
        self.d_info = {k: (1, (0.5, 1.5)) for k, v in self.d_info.items()}
        self.d_init = {k: v[0] for k, v in self.d_info.items()}
            
        return None
        
    def nested_schur_decomposition(self, debug=False):
        """This is the outer problem controlled by a trust region solver 
        running on scipy. This is the only method that the user needs to 
        call after the NSD instance is initialized.
        
        Returns:
            results (scipy.optimize.optimize.OptimizeResult): The results from the 
                trust region optimation (outer problem)
                
            opt_dict (dict): Information obtained in each iteration (use for
                debugging)
                
        """    
        global param_dict
        
        print(iteration_spacer.substitute(iter='NSD Start'))
    
        print(self.d_init)
    
        d_init = self.d_init
        d_bounds = self._generate_bounds_object()
    
        self.d_iter = list()
        def callback(x, *args):
            self.d_iter.append(x)
    
        if self.method in ['trust-exact', 'trust-constr']:

            fun = _inner_problem
            x0 = list(d_init.values())
            args = (self.models_dict,)
            jac = _calculate_m
            hess = _calculate_M
            
            tr_options={
                'xtol': 1e-6,
                }
            
            callback(x0)
            results = minimize(fun, 
                               x0, 
                               args=args, 
                               method=self.method,
                               jac=jac,
                               hess=hess,
                               callback=callback,
                               bounds=d_bounds,
                               options=tr_options,
                           )
            self.parameters_opt = {k: results.x[i]*self.d_init_unscaled[k] for i, k in enumerate(self.d_init.keys())}
            
        if self.method in ['newton']:
            x0 = list(d_init.values())
            results = self._run_newton_step(x0, self.models_dict)
            self.parameters_opt = {k: results[i] for i, k in enumerate(self.d_init.keys())}
        
        if debug:
            return results, param_dict
        else:
            return results
        
    def plot_paths(self, filename):
        
        global param_dict
        
        df_p = pd.DataFrame(param_dict).T
        print(df_p)
        p_max = df_p.max()
        p_min = df_p.min()
        
        df_p = df_p.drop(index=[0])
        
        line_options = {
                        'linewidth' : 3,
                        'alpha' : 0.8,
                        'marker':'o',
                        'markersize' : 4,
                        }
        
        if not self.use_scaling:
            df_norm = (df_p-p_min)/(p_max-p_min)
        else:
            df_norm = df_p
            # print(df_norm)
            # for i, row in df_p.iterrows():
            #     df_norm.iloc[i, :] = df_norm.iloc[i, :]/df_norm.iloc[i, 0]
        
        print(df_norm)
        fig = plt.figure(figsize=(12.0, 8.0)) # in inches!
        
        for i, k in enumerate(self.parameters_opt.keys()):
            plt.plot(df_norm.index, df_norm.loc[:,i], label=k, **line_options)
        plt.legend(fontsize=20, ncol=5)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Scaled Parameter Value', fontsize=20)
        plt.savefig(filename + '.png', dpi=600)

        
        return None

    
    def _run_newton_step(self, d_init, models):
        """This runs a basic Newton step algorithm - use a decent alpha!
        
        UNDER CONSTRUCTION!
        """
        tol = 1e-6
        alpha = 0.4
        max_iter = 40
        counter = 0
        self.d_iter.append(d_init)
        
        while True:   
        
            _inner_problem(d_init, models, generate_gradients=False)
            M = opt_dict[opt_count]['M']
            m = opt_dict[opt_count]['m']
            d_step = np.linalg.inv(M).dot(-m)
            d_init = [d_init[i] + alpha*d_step[i] for i, v in enumerate(d_init)]
            self.d_iter.append(d_init)
            
            if max(d_step) <= tol:
                
                print(f'Terminating sequence: minimum tolerance in step size reached ({tol}).')
                break
            
            if counter == max_iter:
                print(f'Terminating sequence: maximum number of iterations reached ({max_iter})')
                break
            
            counter += 1
            
        return d_init
                 
def _optimize(model, d_vals, verbose=False):
    """Solves the optimization problem with optional k_aug sensitivity
    calculations (needed for Nested Schur Decomposition)
    
    Args:
        model (pyomo ConcreteModel): The current model used in parameter
            fitting
            
        d_vals (dict): The dict of global parameter values
        
        verbose (bool): Defaults to false, option to see solver output
        
    Returns:
        model (pyomo ConcreteModel): The model after optimization
        
    """
    global global_param_name
    global parameter_var_name
    
    ipopt = SolverFactory('ipopt')
    tmpfile_i = "ipopt_output"

    for k, v in getattr(model, parameter_var_name).items():
        getattr(model, parameter_var_name)[k].unfix()

    for k, v in getattr(model, global_param_name).items():
        getattr(model, global_param_name)[k] = d_vals[k]
    
    results = ipopt.solve(model,
                          symbolic_solver_labels=True,
                          keepfiles=False, 
                          tee=verbose, 
                          logfile=tmpfile_i)
    
    return model


def _scenario_optimization(k_model, model, parameter_var_name, d_init):
    
    valid_parameters_scenario = pe_sets[k_model] 
    #print(valid_parameters_scenario)
    valid_parameters = dict(getattr(model, parameter_var_name).items()).keys()
    model_opt = _optimize(model, d_init)
    
    # Possible speed up here - return sparse instead of df - find location of constraints
    kkt_df, var_ind, con_ind_new = _get_kkt_matrix(model_opt)
    duals = {key: model_opt.dual[getattr(model_opt, global_constraint_name)[key]] for key, val in getattr(model_opt, global_param_name).items()}
    col_ind  = [var_ind.loc[var_ind[0] == f'{parameter_var_name}[{v}]'].index[0] for v in valid_parameters]
    dummy_constraints = _get_dummy_constraints(model_opt)
    dc = [d for d in dummy_constraints]
    
    use_SOLE = True
    
    # Perform the calculations to get M and m
    K = kkt_df.drop(index=dc, columns=dc)
    E = np.zeros((len(dummy_constraints), K.shape[1]))
    
    for i, indx in enumerate(col_ind):
        E[i, indx] = 1
  
    # Get S inverse
    if not use_SOLE:
      
        # Solve by inverting matrices using np.linalg.inv
        K_i_inv = pd.DataFrame(np.linalg.inv(K.values), index=K.index, columns=K.columns)
        P = E.dot(K_i_inv.values).dot(E.T)
        Si = np.linalg.inv(P)
  
    elif use_SOLE and not use_conditioning:
        
        # Make square matrix (A) of Eq. 14
        top = (K, E.T)
        bot = (E, np.zeros((len(dummy_constraints), len(dummy_constraints))))
    
        top = np.hstack(top)
        bot = np.hstack(bot)
        A = np.vstack((top, bot))
    
        # Make the rhs (b) of Eq. 14
        b = np.vstack((np.zeros((K.shape[0], len(dummy_constraints))), -1*np.eye(len(dummy_constraints))))
    
        # Solve for Qi and Si
        rhs = np.linalg.solve(A,b)
        Si = rhs[-rhs.shape[1]:, :]
    
    elif use_SOLE and use_conditioning:
        
        # Make square matrix (A) of Eq. 16 to solve for P_inv
        top = (K, E.T)
        bot = (E, np.zeros((len(dummy_constraints), len(dummy_constraints))))
    
        top = np.hstack(top)
        bot = np.hstack(bot)
        A = np.vstack((top, bot))
    
        # Make the rhs (b) of Eq. 16 - this is R1 = 0, R2 = -I, as above
        b = np.vstack((np.zeros((K.shape[0], len(dummy_constraints))), -1*np.eye(len(dummy_constraints))))
    
        # Solve for Qi and Si
        rhs = np.linalg.solve(A,b)
        P_inv = rhs[-rhs.shape[1]:, :]
        Si = P_inv
        Si -= Q_value*np.eye(Si.shape[0])
    
    Mi = pd.DataFrame(Si, index=valid_parameters, columns=valid_parameters)
    
    if not cross_updating:
        parameters_not_to_update = set(valid_parameters).difference(set(valid_parameters_scenario))
        Mi = Mi.drop(index=list(parameters_not_to_update), columns=list(parameters_not_to_update))
   
    return Mi, duals, model_opt.objective.expr()

def _inner_problem(d_init_list, models, generate_gradients=False, initialize=False):
    """Calculates the inner problem using the scenario info and the global
    parameters d
    
    Args:
        d_init_last (list): list of the parameter values
        
        models (dict): the dict of pyomo models used as supplemental args
        
        generate_gradients (bool): If the d values do not line up with the 
            current iteration (if M or m is calculated before the inner prob),
            then the inner problem is solved to generate the corrent info
            
        initialize (bool): Option only used in the initial optimization before 
            starting the NSD routine
    
    Returns:
        Either returns None (generating gradients) or the scalar value of the 
        sum of objective function values from the inner problems
        
    """    
    global opt_count
    global opt_dict
    global global_constraint_name
    global parameter_var_name
    global global_param_name
    global parameter_names
    global pe_sets
    global param_dict
    global cross_updating
    global Q_value
    global use_conditioning
    global use_mp
    
    opt_count += 1      
    options = {'verbose' : False}
    _models = copy.copy(models) 
    
    verbose = False
    
    Si = []
    Ki = []
    Ei = []
    M_size = len(d_init_list)
    M = pd.DataFrame(np.zeros((M_size, M_size)), index=parameter_names, columns=parameter_names)
    m = pd.DataFrame(np.zeros((M_size, 1)), index=parameter_names, columns=['dual'])
    objective_value = 0
    
    d_init = dict(zip(parameter_names, d_init_list))
    
    if opt_count == 0:
        print(iteration_spacer.substitute(iter=f'Inner Problem Initialization'))
        if verbose:
            print(f'Initial parameter set: {d_init}')
    else:
        
        if verbose:
            print(iteration_spacer.substitute(iter=f'Inner Problem {opt_count}'))
            print(f'Current parameter set: {d_init}')
    
    if use_mp:
    
        model_data = list(zip(models.keys(), models.values(), [parameter_var_name]*len(_models), [d_init]*len(_models)))

        with mp.Pool(num_cpus) as pool:
            results = pool.starmap(_scenario_optimization, model_data)
        
        for i, res in enumerate(results):
        
            Mi = res[0]
            duals = res[1]
            obj_val = res[2]
        
            M = M.add(Mi).combine_first(M)
            M = M[parameter_names]
            M = M.reindex(parameter_names)
            eig, u = np.linalg.eigh(M)
            condition = max(abs(eig))/min(abs(eig))      
            if verbose:
                print(f'M conditioning: {condition}')
            
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]
            
            objective_value += obj_val
        
    else:
    
        for k_model, model in _models.items():
            if verbose:
                print(f'Performing inner optimization: {k_model}')
            Mi, duals, obj_val = _scenario_optimization(k_model, model, parameter_var_name, d_init)
            
            M = M.add(Mi).combine_first(M)
            M = M[parameter_names]
            M = M.reindex(parameter_names)
            eig, u = np.linalg.eigh(M)
            condition = max(abs(eig))/min(abs(eig))      
            if verbose:
                print(f'M conditioning: {condition}')
            
            for param in m.index:
                if param in duals.keys():
                    m.loc[param] = m.loc[param] + duals[param]
            
            objective_value += obj_val
            

    if divmod(opt_count, 10)[1] == 0:
        print('\nIteration\t\tObjective\t\t\tAbs Diff\n')
    if opt_count <= 1:
        obj_diff = 0
    else:
        obj_diff = objective_value - opt_dict[opt_count-1]['obj']
    
    if opt_count > 0:
        print(f'{opt_count}\t\t\t\t{objective_value:e}\t\t{abs(obj_diff):e}') 
    opt_dict[opt_count] = { 'd': d_init_list,
                            'obj': objective_value,
                            'M': M.values,
                            'm': m.values.ravel(),
                            } 
    
    param_dict[opt_count] = d_init_list
    
    if len(opt_dict) > 1:
        del opt_dict[opt_count - 1]
    if not generate_gradients:
        return objective_value
    else:
        return None

def _get_kkt_matrix(model):
    """This uses pynumero to get the Hessian and Jacobian in order to build the
    KKT matrix
    
    Args:
        model (pyomo ConcreteModel): the current model used in the inner 
            problem optimization
            
    Returns:
        KKT (pandas.DataFrame): KKT matrix for the current iteration
        
        var_index_names (list): list of variable names
        
        con_index_names (list): list of constraint names
    
    """
    nlp = PyomoNLP(model)
    varList = nlp.get_pyomo_variables()
    conList = nlp.get_pyomo_constraints()
    duals = nlp.get_duals()
    
    J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
    H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
    
    var_index_names = [v.name for v in varList]
    con_index_names = [v.name for v in conList]

    J_df = pd.DataFrame(J.todense(), columns=var_index_names, index=con_index_names)
    H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
    
    var_index_names = pd.DataFrame(var_index_names)
    
    KKT_up = pd.merge(H_df, J_df.transpose(), left_index=True, right_index=True)
    KKT = pd.concat((KKT_up, J_df))
    KKT = KKT.fillna(0)
    
    return KKT, var_index_names, con_index_names

def _get_dummy_constraints(model):
    """Get the locations of the contraints for the local and global parameters
    
    Args:
        models (pyomo ConcreteModel): The current model in the inner problem
        
    Returns:
        dummy_constraints (str): the names of the dummy contraints for the 
            parameters
    
    """
    global global_constraint_name
    global parameter_var_name
    global global_param_name
    
    dummy_constraint_name = global_constraint_name
    dummy_constraint_template = Template(f'{dummy_constraint_name}[$param]')
    parameters = getattr(model, global_param_name).keys()
    dummy_constraints = [dummy_constraint_template.substitute(param=k) for k in parameters]
    
    return dummy_constraints

def _calculate_M(x, scenarios):
    """Calculates the Hessian, M
    This is scipy.optimize.minimize conform
    Checks that the correct data is retrieved
    Needs the global dict to get information from the inner optimization
    
    Args:
        x (list): current parameter values
        
        scenarios (dict): The dict of scenario models
        
    Returns:
        M (np.ndarray): The M matrix from the NSD method
    
    """
    global opt_dict
    global opt_count
    
    if opt_count == 0 or any(opt_dict[opt_count]['d'] != x):
        _inner_problem(x, scenarios, generate_gradients=True)

    M = opt_dict[opt_count]['M']
    return M
    
def _calculate_m(x, scenarios):
    """Calculates the jacobian, m
    This is scipy.optimize.minimize conform
    Checks that the correct data is retrieved
    Needs the global dict to get information from the inner optimization
    
    Args:
        x (list): current parameter values
        
        scenarios (dict): The dict of scenario models
        
    Returns:
        m (np.ndarray): The m matrix from the NSD method
    
    """
    global opt_dict
    global opt_count
    
    if opt_count == 0 or any(opt_dict[opt_count]['d'] != x):
        _inner_problem(x, scenarios, generate_gradients=True)
    
    m = opt_dict[opt_count]['m']
    return m    
    
class ScalingVisitor(EXPR.ExpressionReplacementVisitor):

    def __init__(self, scale):
        super(ScalingVisitor, self).__init__()
        self.scale = scale

    def visiting_potential_leaf(self, node):
      
        if node.__class__ in native_numeric_types:
            return True, node

        if node.is_variable_type():
           
            return True, self.scale[id(node)]*node

        if isinstance(node, EXPR.LinearExpression):
            node_ = copy.deepcopy(node)
            node_.constant = node.constant
            node_.linear_vars = copy.copy(node.linear_vars)
            node_.linear_coefs = []
            for i,v in enumerate(node.linear_vars):
                node_.linear_coefs.append( node.linear_coefs[i]*self.scale[id(v)] )
            return True, node_

        return False, None