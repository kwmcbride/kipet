"""
Kipet: Kinetic parameter estimation toolkit
Copyright (c) 2016 Eli Lilly.
 
Example from Chen and Biegler, Reduced Hessian Based Parameter Selection and
    Estimation with Simultaneous Collocation Approach (AIChE 2020) paper with
    a CSTR for a simple reaction.
    
This code generates three examples used as experiments for the NSD method. It
is also recommended to use the coupled EstimationPotential method with it as
well.

Note: sometimes the kernel dies when using the full model
"""
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyomo.core as pyomo
from pyomo.environ import ( 
    Objective,
    exp,
    ) 

from kipet.library.data_tools import add_noise_to_signal
from kipet.library.EstimationPotential import EstimationPotential
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import (
        TemplateBuilder,
        Component,
        KineticParameter,
        )

def rule_objective(model):
    """This function defines the objective function for the estimability
    
    This is equation 5 from Chen and Biegler 2020. It has the following
    form:
        
    .. math::
        \min J = \frac{1}{2}(\mathbf{w}_m - \mathbf{w})^T V_{\mathbf{w}}^{-1}(\mathbf{w}_m - \mathbf{w})
        
    Originally KIPET was designed to only consider concentration data in
    the estimability, but this version now includes complementary states
    such as reactor and cooling temperatures. If complementary state data
    is included in the model, it is detected and included in the objective
    function.
    
    Args:
        model (pyomo.core.base.PyomoModel.ConcreteModel): This is the pyomo
        model instance for the estimability problem.
            
    Returns:
        obj (pyomo.environ.Objective): This returns the objective function
        for the estimability optimization.
    
    """
    obj = 0

    for k in set(model.mixture_components.value_list) & set(model.measured_data.value_list):
        for t, v in model.C.items():
            obj += 0.5*(model.C[t] - model.Z[t]) ** 2 / model.sigma[k]**2
    
    for k in set(model.complementary_states.value_list) & set(model.measured_data.value_list):
        for t, v in model.U.items():
            obj += 0.5*(model.X[t] - model.U[t]) ** 2 / model.sigma[k]**2      

    return Objective(expr=obj)

def check_discretization(model, ncp=3, nfe=50):
        """Checks is the model is discretized and discretizes it in the case
        that it is not
        
        Args:
            model (ConcreteModel): A pyomo ConcreteModel
            
            ncp (int): number of collocation points used
            
            nfe (int): number of finite elements used
            
        Returns:
            None
            
        """
        if not model.alltime.get_discretization_info():
        
            model_pe = ParameterEstimator(model)
            model_pe.apply_discretization('dae.collocation',
                                            ncp = ncp,
                                            nfe = nfe,
                                            scheme = 'LAGRANGE-RADAU')
        
        return None

def run_simulation(simulator, nfe=50, ncp=3, use_only_FE=True):
    """This is not necessary, but is used for generating data used in the
    estimation algorithm
    """
    simulator.apply_discretization('dae.collocation',
                                   ncp = ncp,
                                   nfe = nfe,
                                   scheme = 'LAGRANGE-RADAU')

    options = {'solver_opts' : dict(linear_solver='ma57')}
    
    results_pyomo = simulator.run_sim('ipopt_sens',
                                      tee=False,
                                      solver_options=options,
                                      )
    
    Z_data = pd.DataFrame(results_pyomo.Z)
    try:
        X_data = pd.DataFrame(results_pyomo.X)
    except:
        pass
    
    use_X_data = True if X_data.size > 0 else False
    
    if use_only_FE:
        
        t = np.linspace(0, ncp*nfe, nfe+1).astype(int)
        
        Z_data = Z_data.iloc[t, :]
        Z_data.drop(index=0, inplace=True)
        
        if use_X_data:
            X_data = X_data.iloc[t, :]
            X_data.drop(index=0, inplace=True)
        
    if not use_X_data:
        X_data = results_pyomo.X
        
    return Z_data, X_data, results_pyomo

def make_model_dict():

    models = {}    

    components = [
            Component('A', 1000, 0.1),
            ]
        
    parameters = [
            KineticParameter('Tf',   (283.15, 400), 293.15, 0.09),
            KineticParameter('Cfa',  (0, 5000), 2500, 0.01),
            KineticParameter('rho',  (100, 2000), 1025, 0.01),
            KineticParameter('delH', (0.0, 400), 160, 0.01),
            KineticParameter('ER',   (0.0, 5000), 255, 0.01),
            KineticParameter('k',    (0.0, 10), 2.5, 0.01),
            KineticParameter('Tfc',  (283.15, 300), 283.15, 0.01),
            KineticParameter('rhoc', (0.0, 2000), 1000, 0.01),
            KineticParameter('h',    (0.0, 5000), 3600, 0.01),
            ]
    
    constants = {
            'F' : 0.1, # m^3/h
            'Fc' : 0.15, # m^3/h
            'Ca0' : 1000, # mol/m^3
            'V' : 0.2, # m^3
            'Vc' : 0.055, # m^3
            'A' : 4.5, # m^2
            'Cpc' : 1.2, # kJ/kg/K
            'Cp' : 1.55, #6 kJ/kg/K
            }
    
    # Make it easier to use the constants
    C = constants
    
    noise = {
            'A' : 1, #0.01,
            'T' : 1, #0.25,
            }
    
    def simulation_reactor():
        """This prepares the simulated data for the CSTR examples in the first
        and second experiments. This is the CSTR problem found in the 2020
        paper by Chen and Biegler.
        
        """
        builder = TemplateBuilder()
    
        builder.add_complementary_state_variable('T',  293.15)
        builder.add_complementary_state_variable('Tc', 293.15)
    
        # Prepare components
        for com in components:
            builder.add_mixture_component(com.name, com.init)
        
        # Prepare parameters
        for param in parameters:
            builder.add_parameter(param.name, param.init)
          
        
        builder.set_odes_rule(rule_odes)
        times = (0.0, 5.0)
        builder.set_model_times(times)
        pyomo_model = builder.create_pyomo_model()
        simulator = PyomoSimulator(pyomo_model)
        Z_data, X_data, results = run_simulation(simulator)
    
        return Z_data, X_data, results
    
    def rule_odes(m,t):
        """ODE system of the CSTR"""
        
        Ra = m.P['k']*pyomo.exp(-m.P['ER']/m.X[t,'T'])*m.Z[t,'A']
        exprs = dict()
        exprs['A'] = C['F']/C['V']*(m.P['Cfa']-m.Z[t,'A']) - Ra
        exprs['T'] = C['F']/C['V']*(m.P['Tf']-m.X[t,'T']) + m.P['delH']/(m.P['rho'])/C['Cp']*Ra - m.P['h']*C['A']/(m.P['rho'])/C['Cp']/C['V']*(m.X[t,'T'] - m.X[t,'Tc'])
        exprs['Tc'] = C['Fc']/C['Vc']*(m.P['Tfc']-m.X[t,'Tc']) + m.P['h']*C['A']/(m.P['rhoc'])/C['Cpc']/C['Vc']*(m.X[t,'T'] - m.X[t,'Tc'])
        return exprs

    def simulation_lab_reaction():
        """This prepares the simulated data for the simple isothermal lab-scale
        reaction used in the third experiment
        
        """
        builder = TemplateBuilder()
        builder.add_mixture_component({'A': 1000})
        factor = 1.0
        
        for param in parameters:
            if param.name in ['k', 'ER']:
                builder.add_parameter(param.name, param.init*factor)
    
        builder.set_odes_rule(rule_odes_conc)
        times = (0.0, 2.0)
        builder.set_model_times(times)
        pyomo_model = builder.create_pyomo_model()
        simulator = PyomoSimulator(pyomo_model)
        Z_data, X_data, results = run_simulation(simulator)
        
        # Some random measurement locations
        conc_measurement_index = [1, 2]
        Z_data = results.Z.iloc[conc_measurement_index, :]
        
        return Z_data
    
    def rule_odes_conc(m,t):
        """ODE system for the isothermal lab-scale reaction"""
        
        Ra = m.P['k']*pyomo.exp(-m.P['ER']/315)*m.Z[t,'A']
        exprs = dict()
        exprs['A'] = -Ra
        return exprs
    
    def make_exp_1():
        """ Add an experiment (1) 
        
        This is the example used in the paper by Chen and Biegler with 50
        temperature measurements
        
        """
        builder = TemplateBuilder() 
        factor = 1.2
        
        builder.add_complementary_state_variable('T',  293.15)
        builder.add_complementary_state_variable('Tc', 293.15)
        builder.set_odes_rule(rule_odes)
    
        for com in components:
            builder.add_mixture_component(com.name, com.init)
        
        for param in parameters:
            builder.add_parameter(param.name, bounds=param.bounds, init=param.init*factor)
       
        times = (0.0, 5.0)
        builder.set_model_times(times)
        Z_data, X_data, results = simulation_reactor()
        
        #X_data['T'] = add_noise_to_signal(X_data['T'], noise['T'])
        builder.add_complementary_states_data(pd.DataFrame(X_data['T']))
        
        conc_measurement_index = [7, 57, 99]
        Z_data = results.Z.iloc[conc_measurement_index, :]
        #Z_data['A'] = add_noise_to_signal(Z_data['A'], noise['A'])
        builder.add_concentration_data(pd.DataFrame(Z_data))
        
        model = builder.create_pyomo_model()
        
        return model
    
    def make_exp_2():
        """ Add an experiment (2) 
        
        This uses only 3 conc measurements at the same conditions as the first
        
        """
        builder = TemplateBuilder()
        factor = 1.2
        
        builder.add_complementary_state_variable('T',  293.15)
        builder.add_complementary_state_variable('Tc', 293.15)
        builder.set_odes_rule(rule_odes)
        
        for com in components:
            builder.add_mixture_component(com.name, com.init)
        
        for param in parameters:
            builder.add_parameter(param.name, bounds=param.bounds, init=param.init*factor)
           
        times = (0.0, 5.0)
        builder.set_model_times(times)
        
        Z_data, X_data, results = simulation_reactor()
        
        conc_measurement_index = [7, 20, 50, 57, 70, 80, 90, 99, 120, 140]
        Z_data = results.Z.iloc[conc_measurement_index, :]
        Z_data['A'] = add_noise_to_signal(Z_data['A'], noise['A'])
        builder.add_concentration_data(pd.DataFrame(Z_data))
        model = builder.create_pyomo_model()
        
        return model
    
    def make_exp_3():
        """ Add an experiment (3) 
        
        This is an isothermal (315 K) lab-scale reaction (no CSTR, no feed)
        
        """
        builder = TemplateBuilder()
        factor = 0.9
        
        builder.set_odes_rule(rule_odes_conc)
        components = {'A': 1000}
        builder.add_mixture_component(components)
        
        for param in parameters:
            if param.name in ['k', 'ER']:
                builder.add_parameter(param.name, bounds=param.bounds, init=param.init*factor)
       
        times = (0.0, 2.0)
        builder.set_model_times(times)
        Z_data = simulation_lab_reaction()
        #Z_data['A'] = add_noise_to_signal(Z_data['A'], 0.01)
        builder.add_concentration_data(pd.DataFrame(Z_data))
        model = builder.create_pyomo_model()
        
        return model
    
    """ Discretize and and add an objective function"""
    
    models['Exp-1'] = make_exp_1()
    #models['Exp-2'] = make_exp_2()
    models['Exp-3'] = make_exp_3()
    
    for k, model in models.items():
        
        check_discretization(model)
        model.objective = rule_objective(model)
        
    return models, parameters