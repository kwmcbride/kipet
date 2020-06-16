"""
Multiple Scenarios using a nested Schur decomposition strategy (models)
This is for the stand-alone version of NSD

To run this example, you need to have the latest version of Kipet.
"""
import copy
import sys
import numpy as np
import pandas as pd

from pyomo.environ import ( 
    Objective,
    ) 

from kipet.library.data_tools import add_noise_to_signal
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.TemplateBuilder import TemplateBuilder

from NestedSchurDecomposition import NestedSchurDecomposition as NSD   

# Helper functions for setting up the objective, the model discretization, and
# for generating some pseudo experimental data

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

    for k in model.mixture_components & model.measured_data:
        for t, v in model.C.items():
            obj += 0.5*(model.C[t] - model.Z[t]) ** 2 / model.sigma[k]**2
    
    for k in model.complementary_states & model.measured_data:
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
    
    for k, v in simulator.model.P.items():
        simulator.model.P[k].fix()
        
    results_pyomo = simulator.run_sim('ipopt_sens',
                                      tee=False,
                                      solver_options=options,
                                      )

    if with_plots:
        results_pyomo.Z.plot.line(legend=True)
        plt.xlabel("time (h)")
        plt.ylabel("Concentration (mol/L)")
        plt.title("Concentration Profile")
        
        plt.show()
    
    Z_data = pd.DataFrame(results_pyomo.Z)
    X_data = None
    
    if use_only_FE:
        
        t = np.linspace(0, ncp*nfe, nfe+1).astype(int)
        Z_data = Z_data.iloc[t, :]
        Z_data.drop(index=0, inplace=True)

        
    return Z_data, X_data, results_pyomo

if __name__ == "__main__":

    with_plots = False
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    # Set up the problem - here are the odes to describe the reaction
 
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    
    # This is to test the robustness - A to B, no C
    def rule_odes_2(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']#-m.P['k2']*m.Z[t,'B']
        #exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    times = (0.0, 10.0)

    # Set up the scenarios - this needs to be made easier for the user
    # Each case is made using simulated data with noise added and with
    # different initial conditions (hence the four experiments)
    # Note: the fourth experiment has only one reaction and one parameter
    
    models = {}
    
    # Declare all global variables used in simulation
    d_init = {'k1' : (2.5, (0.0, 5.0)),
              'k2' : (0.8, (0.0, 1.0)),
              }
    
    # Declare all global variables as the initial guess
    d_init_guess = {'k1' : (2.5, (0.0, 5.0)),
                    'k2' : (0.5, (0.0, 1.0)),
                   }


    # Scenario 1 #############################################################
    builder = TemplateBuilder()
    build = builder    
    components = {'A':1e-3,'B':0,'C':0}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
        
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 1 
    pyomo_model = build.create_pyomo_model()
    simulator = PyomoSimulator(pyomo_model)
    Z_data, X_data, results = run_simulation(simulator)
    
    conc_measurement_index = [7, 57, 99]
    Z_data = results.Z.iloc[conc_measurement_index, :]
    #Z_data = add_noise_to_signal(Z_data, 1e-5)
    
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    build.add_concentration_data(Z_data)
    model = build.create_pyomo_model()
    models[0] = model
    
    # Scenario 2 #############################################################
    builder2 = TemplateBuilder()    
    build = builder2
    components = {'A':0.5e-3,'B':1e-4,'C':0}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
   
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 2 
    pyomo_model2 = build.create_pyomo_model()
    simulator2 = PyomoSimulator(pyomo_model2)
    Z_data2, X_data2, results2 = run_simulation(simulator2)
    
    conc_measurement_index2 = [25, 28, 80]
    Z_data2 = results2.Z.iloc[conc_measurement_index2, :]
    #Z_data2 = add_noise_to_signal(Z_data2, 1e-5)
    
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    build.add_concentration_data(Z_data2)
    model = build.create_pyomo_model()
    models[1] = model
    
    # Scenario 3 #############################################################
    builder3 = TemplateBuilder()   
    build = builder3
    components = {'A':1.5e-3,'B':4e-4,'C':1e-4}
    build.add_mixture_component(components)
    
    for d, dv in d_init.items():
        build.add_parameter(d, init=dv[0], bounds=dv[1])
    
    build.set_odes_rule(rule_odes)
    build.set_model_times(times)
    
    # Simulate data for scenario 3
    pyomo_model3 = build.create_pyomo_model()
    simulator3 = PyomoSimulator(pyomo_model3)
    Z_data3, X_data3, results3 = run_simulation(simulator3)
    
    conc_measurement_index3 = [12, 30, 50, 70, 82, 140]
    Z_data3 = results3.Z.iloc[conc_measurement_index3, :]
    #Z_data3 = add_noise_to_signal(Z_data3, 1e-5)
    
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    build.add_concentration_data(Z_data3)
    model = build.create_pyomo_model()
    models[2] = model
    
    # Scenario 4 #############################################################
    builder4 = TemplateBuilder()   
    build = builder4
    components = {'A':1.5e-3,'B':4e-4}
    build.add_mixture_component(components)
    build.add_parameter('k1', init=d_init['k1'][0], bounds=d_init['k1'][1])
    build.set_odes_rule(rule_odes_2)
    build.set_model_times(times)
    
    # Simulate data for scenario 4
    pyomo_model4 = build.create_pyomo_model()
    simulator4 = PyomoSimulator(pyomo_model4)
    Z_data4, X_data4, results4 = run_simulation(simulator4)
    
    conc_measurement_index4 = [12, 30, 50, 70, 82, 140]
    Z_data4 = results4.Z.iloc[conc_measurement_index4, :]
    #Z_data4 = add_noise_to_signal(Z_data4, 1e-5)
    
    build._parameters_init = {k: v[0] for k, v in zip(build._parameters.keys(), d_init_guess.values())}
    build.add_concentration_data(Z_data4)
    model = build.create_pyomo_model()
    models[3] = model
    
    for k, model in models.items():
        check_discretization(model)
        model.objective = rule_objective(model)
    
    # Perform the NSD parameter optimization
    
    # Specify the parameter name in your model (required!)
    options = {
               'method': 'trust-constr',
               }
    
    parameter_var_name = 'P'
    nsd = NSD(models, d_init_guess, parameter_var_name, options)
    results = nsd.nested_schur_decomposition()
    
    print(f'\nThe final parameter values are:\n{nsd.parameters_opt}')