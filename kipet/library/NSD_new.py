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
import os
import psutil
from string import Template
import sys

from pyomo.environ import *
from scipy.sparse import csc_matrix, coo_matrix
from kipet.library.nsd_funs.ip_line_search import ip_line_search
import pandas as pd
import sys


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
    Var,
    )

from pyomo.core.expr import current as EXPR

from scipy.optimize import (
    Bounds,
    minimize,
    )

global opt_dict
opt_dict = dict()
global opt_count
opt_count = 0
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
        #self._orig_models = copy.deepcopy(models)
        self.models_dict = copy.deepcopy(models)
        
        # The global parameter information is needed, especially the bounds
        self.d_info = copy.copy(d_info)
        # This may be redundant if you just want to extract the init values
        self.d_init = {k: v[0] for k, v in d_info.items()}
        
        # Arrange the kwargs
        self._kwargs = {} if kwargs is None else copy.copy(kwargs)
        
        # Options - inner problem optimization
        #self.ncp = self._kwargs.pop('ncp', 3)
        #self.nfe = self._kwargs.pop('nfe', 50)
        self.verbose = self._kwargs.pop('verbose', False)
        #self.sens = self._kwargs.pop('use_k_aug', True)
        #self.objective_name = self._kwargs.pop('objective_name', None)
        #self.gtol = self._kwargs.pop('gtol', 1e-12)
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
        
        all_params = set()
        for exp in self.pe_sets.values():
            all_params = all_params.union({k for k in exp})
    
        self.num_params = len(all_params)
    
        # Run assertions that the model is correctly structured
        self._test_models()
        
        self.d_init_unscaled = self.d_init
        if self.use_scaling:
            self._scale_models()
        
        
        global mu_target
        mu_target = 1e-1
        mu_init = 0.1
        accept_tol = 2e-7
        accept_iter = 3
        tol = mu_init

        with open('ipopt.opt', 'w') as f:
            f.write('mu_target '+str(mu_target)+'\n')
            f.write('mu_init '+str(mu_init)+'\n')
            f.write('output_file temp\n')
            f.write('tol ' + str(tol) + '\n')
            f.write('linear_solver ma27\n')
            f.write('max_iter 25\n')
            f.write('print_user_options yes\n')
            f.write('print_info_string yes\n')
            # f.write('compl_inf_tol '+str(compl_tol)+'\n')
            f.close()
        
        # Add the global constraints to the model
        self._add_global_constraints()
        self._prep_models()

        #self._add_barrier_terms(mu=1e-1)

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
        
        """Adds the conditioning terms to the objective function"""
        
        global global_param_name
        
        for model in self.models_dict.values():
            for key, param in model.P.items():
            
                Q_term = self.conditioning_Q*(model.P[key] - getattr(model, global_param_name)[key])**2
                model.objective.expr += Q_term
                
        return None
    
    def _add_barrier_terms(self, mu=1e4):
        
        """Adds the conditioning terms to the objective function"""
        
        for model in self.models_dict.values():
            
            model.mu = Var(initialize=mu)
            model.mu.fix()
            #for key, param in model.P.items():
            # Set the global bounds here - they are currently the same in each model
            B_term = model.mu*sum(log(model.P[key].bounds[1] - model.P[key]) for key in model.P)
            B_term += model.mu*sum(log(model.P[key] - model.P[key].bounds[0]) for key in model.P)
            model.objective.expr -= B_term
            
            print(model.objective.expr.to_string())
                
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
        
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float) 
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
                    
                # Add algebraics here - Tom's models need this?
            
                for k, v in model.odes.items(): 
                    scaled_expr = scale_expression(v.body, scale)
                    model.odes[k] = scaled_expr == 0
                    #print(scaled_expr)
                
            return models
        
        def scale_expression(expr, scale):
            
            visitor = ScalingVisitor(scale)
            return visitor.dfs_postorder_stack(expr)
        
        self.models_dict = scale_parameters(models)

        scaled_bounds = {}

        for key, model in self.models_dict.items():
            rho = 10
            for k, v in model.P.items():
                ub = self.d_info[k][1][1]/self.d_init_unscaled[k]
                lb = self.d_info[k][1][0]/self.d_init_unscaled[k]
                
                if ub < 1:
                    print('Bounds issue, pushing upper bound higher')
                    ub = 1.1
                if lb > 1:
                    print('Bounds issue, pushing lower bound lower')
                    lb = 0.9
                
                scaled_bounds[k] = (lb, ub)
                
                model.P[k].setlb(lb)
                model.P[k].setub(ub)
                model.P[k].unfix()
                model.P[k].set_value(1)
            
        self.d_info = {k: (1, scaled_bounds[k]) for k, v in self.d_info.items()}
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
    
        mu_start = 0.1
    
        if self.method in ['trust-exact', 'trust-constr']:

            fun = _inner_problem
            mu = 0
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
            
            if self.use_scaling:
                s_factor = self.d_init_unscaled
            else:
                s_factor = {k: 1 for k in self.d_init.keys()}
            
            self.parameters_opt = {k: results.x[i]*s_factor[k] for i, k in enumerate(self.d_init.keys())}

        # Use this now to build the line search method
        if self.method in ['newton']:
            x0 = d_init
            results = self._run_newton_step(x0, self.models_dict)
            
            if self.use_scaling:
                s_factor = self.d_init_unscaled
            else:
                s_factor = {k: 1 for k in self.d_init.keys()}
                
            self.parameters_opt = {k: results[k]*s_factor[k] for k in results.keys()}

        if self.method in ['ip_line_search']:
            
            fun = _inner_problem # _calculate_obj # 
            x0 = list(d_init.values())
            kappa_e = 1.0
            args = (self.models_dict, kappa_e,)
            grad = _calculate_m
            hess = _calculate_M
            results = ip_line_search(fun, x0, args, grad, hess, d_bounds, callback)
            
            if self.use_scaling:
                s_factor = self.d_init_unscaled
            else:
                s_factor = {k: 1 for k in self.d_init.keys()}
            
            self.parameters_opt = {k: results.x[i]*s_factor[k] for i, k in enumerate(self.d_init.keys())}
            
        return results, self.d_iter# , self.d_bounds
        
        if debug:
            return results, param_dict
        else:
            return results
        
    def plot_paths(self, filename=''):
        
        global param_dict
        
        df_p = pd.DataFrame([p for p in param_dict.values()])
        p_max = df_p.max()
        p_min = df_p.min()
        
        print(df_p)
       # df_p = df_p.drop(index=[0])
        
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

        print(df_norm)

        fig = plt.figure(figsize=(12.0, 8.0)) # in inches!
        
        for i, k in enumerate(self.parameters_opt.keys()):
            plt.plot(df_norm.index, df_norm.iloc[:,i], label=k, **line_options)
        plt.legend(fontsize=20, ncol=5)
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Scaled Parameter Value', fontsize=20)
        plt.savefig(filename + '.png', dpi=600)

        return None
                 
def _optimize(model, d_vals, prefix=None, verbose=False, solver='ipopt'):
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
    
    cdir = os.getcwd()
    logdir = cdir + '/log_line/'
    try:  
        os.mkdir(logdir)  
    except OSError:  
        pass 

    print(logdir)
    print(prefix)
 
    ipopt = SolverFactory('ipopt')
    tmpfile_i = logdir + prefix + "_ipopt_output.log"
    #: write some options for ipopt
    with open('ipopt.opt', 'r') as f:
        data = f.readlines()
        f.close()
    data[2] = 'output_file '+tmpfile_i+'\n'
    with open('ipopt.opt', 'w') as f:
        f.writelines(data)
        f.close() 
    results = ipopt.solve(model,
                        symbolic_solver_labels=True,
                        keepfiles=False, 
                        tee=verbose)
    
    print(results.solver.termination_condition)
    

    for k, v in getattr(model, parameter_var_name).items():
        getattr(model, parameter_var_name)[k].unfix()

    for k, v in getattr(model, global_param_name).items():
        getattr(model, global_param_name)[k] = d_vals[k]
    
    results = ipopt.solve(model,
                          symbolic_solver_labels=True,
                          keepfiles=False, 
                          tee=verbose, 
                      #    solver_options=options,
                          logfile=tmpfile_i)
    
    return model

def _solve_inner_problem(model, d_init):
    
    model_opt = _optimize(model, d_init, prefix='model_stuff')#str(k)+'_'+str(opt_count))
    obj_val = model_opt.objective.expr()
      
    return obj_val

def _inner_problem(d_init_list, mu, models, kappa_e, generate_gradients=False, initialize=False):
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
    global parameter_var_name
    global parameter_names
    global use_mp
    global opt_count
    global param_dict
    
    # Update mu_target inside the NSD
    with open('ipopt.opt', 'r') as f_ipopt:
        optdata = f_ipopt.readlines() # assume mu_targe is written in the first line
        f_ipopt.close()

    str_mu = np.format_float_positional(mu, precision=6, unique=False, fractional=False, trim='k')
    
    tol = kappa_e*mu
    str_tol = np.format_float_positional(tol, precision=6, unique=False, fractional=False, trim='k')
    
    optdata[0] = 'mu_target ' + str_mu + '\n'
    optdata[3] = 'tol ' + str_tol + '\n'

    with open('ipopt.opt', 'w') as f_ipopt:
        f_ipopt.writelines(optdata)
        f_ipopt.close()
    
    _models = copy.copy(models) 

    # Set up the parameter values
    if isinstance(d_init_list, np.matrix):
        d_init_list = np.asarray(d_init_list).ravel()
    d_init = dict(zip(parameter_names, d_init_list))
    
    print(d_init)
    param_dict[opt_count] = d_init
    
    print(param_dict)
    # for k_model, model in models.items():
    #     model.mu.set_value(mu)
    
    # Set up for multiprocessing
    if use_mp:
    
        model_data = list(zip(models.keys(), models.values(), [parameter_var_name]*len(_models), [d_init]*len(_models)))

        with mp.Pool(num_cpus) as pool:
            results = pool.starmap(_solve_inner_problem, model_data)
        
        objective_value = sum(results)
        
    # Standard loop
    else:
        objective_value = 0
        for k_model, model in models.items():
            model_opt = _optimize(model, d_init, prefix=f'Iter-{opt_count}-Model-{k_model}')
            objective_value += model_opt.objective.expr()
            
    opt_count += 1
            
    return objective_value    
    
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
    
    sigma_L = []
    sigma_U = []

    for v in varList:
        if v in model.ipopt_zL_out.keys():
            sl = v.value - v.lb
            if sl < 1e-40:
                print('on lower', v.name)
                sl = sl + 10*1e-16*max(1, abs(v.lb))
            sigma_L.append(model.ipopt_zL_out[v]/sl)
        else:
            sigma_L.append(0.0)
        
        if v in model.ipopt_zU_out.keys():
            su = v.ub - v.value
            if su < 1e-40:
                print('on upper', v.name)
                su = su - 10*1e-16*max(1, abs(v.ub))
            sigma_U.append(model.ipopt_zU_out[v]/su)
        else:
            sigma_U.append(0.0)

    SL = np.diag(sigma_L)
    SU = np.diag(sigma_U)
    _SL = coo_matrix(SL)
    _SU = coo_matrix(SU)

    H = H + _SL + _SU
    #print(sigma_L)
    #print(sigma_U)
    
    
    var_index_names = [v.name for v in varList]
    con_index_names = [v.name for v in conList]

    J_df = pd.DataFrame(J.todense(), columns=var_index_names, index=con_index_names)
    H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
    
    var_index_names = pd.DataFrame(var_index_names)
    
    KKT_up = pd.merge(H_df, J_df.transpose(), left_index=True, right_index=True)
    KKT = pd.concat((KKT_up, J_df))
    KKT = KKT.fillna(0)
    
    condition_check = True
    if condition_check:
        p_count = 0
        n_count = 0
        z_count = 0
        e, v = np.linalg.eig(H.todense())
        for i in range(len(e)):
            if e[i].real > 0:
                p_count += 1
            elif e[i].real < 0:
                n_count += 1
            else:
                z_count += 1

        print('---- H ----')
        print('shape:', H.shape,'rank:', np.linalg.matrix_rank(H.todense()))
        print('p:', p_count, 'n:', n_count, 'z:', z_count)
        # print(e)
        print('---- J ----')
        print('shape:', J.shape,'rank:', np.linalg.matrix_rank(J.todense()))
    
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

def _calculate_M(x, mu, scenarios, ke):
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
    global pe_sets
    global parameter_names
    
    all_params = set()
    for exp in pe_sets.values():
        all_params = all_params.union({k for k in exp})
    
    M_size = len(all_params)
    M = pd.DataFrame(np.zeros((M_size, M_size)), index=parameter_names, columns=parameter_names)
    
    for k_model, model in scenarios.items():
    
        kkt_df, var_ind, con_ind_new = _get_kkt_matrix(model)

        valid_parameters_scenario = pe_sets[k_model] 
        valid_parameters = dict(getattr(model, parameter_var_name).items()).keys()

        col_ind  = [var_ind.loc[var_ind[0] == f'{parameter_var_name}[{v}]'].index[0] for v in valid_parameters]
        dummy_constraints = _get_dummy_constraints(model)
        dc = [d for d in dummy_constraints]
        
        # Perform the calculations to get M and m
        K = kkt_df.drop(index=dc, columns=dc)
        E = np.zeros((len(dummy_constraints), K.shape[1]))
        
        for i, indx in enumerate(col_ind):
            E[i, indx] = 1
            
        if not use_conditioning:
        
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
    
        else:
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
        
        # if not cross_updating:
        #     parameters_not_to_update = set(valid_parameters).difference(set(valid_parameters_scenario))
            # Mi = Mi.drop(index=list(parameters_not_to_update), columns=list(parameters_not_to_update))
   
        M = M.add(Mi).combine_first(M)
        M = M[parameter_names]
        M = M.reindex(parameter_names)
        
        # eig, u = np.linalg.eigh(M)
        # condition = max(abs(eig))/min(abs(eig))      
        # if verbose:
        #     print(f'M conditioning: {condition}')
         
    condition_check = True
    # M matrix condition check
    if condition_check:
        p_count = 0
        n_count = 0
        z_count = 0
        e, v = np.linalg.eig(M.values)
        for i in range(len(e)):
            if e[i].real > 0:
                p_count += 1
            elif e[i].real < 0:
                n_count += 1
            else:
                z_count += 1

        print('M size:',M.shape, 'rank:',np.linalg.matrix_rank(M.values))
        print('p:', p_count, 'n:', n_count, 'z:', z_count)
        
    return M.values #csc_matrix(M.values)

def _JH(model, condition_check = True):
    """
    This extracts the Jacobian and the Hessian of the Lagrangian of the model. Also, the Sigma for the duals.

    Args:
        model (pyomo model): The Jacobian and Hessian are extracted from the model

        condition_check (bool): If True, the eigenvalues and rank for _J and _H are checked.
        
    Returns:
        _J (coo matrix): The Jacobian

        _H (coo matrix): The augmented Hessian (includes Sigma_L, Sigma_U)

        var_index_names (list): The list of variable names for the _J and _H. The index corresponds to the row of _J and _H.

        col_index_name (list): The list of constraint names for the _J. The index corresponds to the column of _J. 

    """
    ## Jacobian and Hessian of the Lagrangian extractor
    nlp = PyomoNLP(model)
    varList = nlp.get_pyomo_variables()
    conList = nlp.get_pyomo_constraints() 
    # duals = (-1)*nlp.get_duals() ## Suffix needs to be defined in your model
    
    _J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
    _H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)

    sigma_L = []
    sigma_U = []

    for v in varList:
        if v in model.ipopt_zL_out.keys():
            sl = v.value - v.lb
            if sl < 1e-40:
                print('on lower', v.name)
                sl = sl + 10*1e-16*max(1, abs(v.lb))
            sigma_L.append(model.ipopt_zL_out[v]/sl)
        else:
            sigma_L.append(0.0)
        
        if v in model.ipopt_zU_out.keys():
            su = v.ub - v.value
            if su < 1e-40:
                print('on upper', v.name)
                su = su - 10*1e-16*max(1, abs(v.ub))
            sigma_U.append(model.ipopt_zU_out[v]/su)
        else:
            sigma_U.append(0.0)

    SL = np.diag(sigma_L)
    SU = np.diag(sigma_U)
    _SL = coo_matrix(SL)
    _SU = coo_matrix(SU)

    _H = _H + _SL + _SU
    print(sigma_L)
    print(sigma_U)
    
    var_index_names = [v.name for v in varList]
    con_index_names = [v.name for v in conList]
    print(var_index_names)
    # df_duals = pd.DataFrame({'duals_pynumero':duals, 'duals_suffix':[model.dual[c] for c in conList]}, index=con_index_names)
    # df_duals.to_csv('duals.csv', index=True, header=True)

    # _J and _H matrix condition check
    if condition_check:
        p_count = 0
        n_count = 0
        z_count = 0
        e, v = np.linalg.eig(_H.todense())
        for i in range(len(e)):
            if e[i].real > 0:
                p_count += 1
            elif e[i].real < 0:
                n_count += 1
            else:
                z_count += 1

        print('---- H ----')
        print('shape:', _H.shape,'rank:', np.linalg.matrix_rank(_H.todense()))
        print('p:', p_count, 'n:', n_count, 'z:', z_count)
        # print(e)
        print('---- J ----')
        print('shape:', _J.shape,'rank:', np.linalg.matrix_rank(_J.todense()))

    return _J, _H, var_index_names, con_index_names #, duals

def _calculate_m(x, mu, scenarios, ke):
    """Calculates the matrix m
    
    Args:
        x (list): current parameter values
    
        mu (float): the barrier multiplier
        
        scenarios (dict): The dict of scenario models
 
        ke (float): kappa_e term       
 
    Returns:
        m (np.ndarray): The m matrix from the NSD method
    
    """
    global global_constraint_name
    global parameter_names
    
    m = pd.DataFrame(np.zeros((len(parameter_names), 1)), index=parameter_names, columns=['dual'])
    
    for model_opt in scenarios.values():
    
        duals = {key: model_opt.dual[getattr(model_opt, global_constraint_name)[key]] for key, val in getattr(model_opt, global_param_name).items()}
    
        for param in m.index:
            if param in duals.keys():
                m.loc[param] = m.loc[param] + duals[param]

    return m.values.ravel()


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