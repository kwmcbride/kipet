#  _________________________________________________________________________
#
#  Kipet: Kinetic parameter estimation toolkit
#  Copyright (c) 2016 Eli Lilly.
#  _________________________________________________________________________

# Sample Problem 
# Estimation with unknow variancesof spectral data using pyomo discretization 
#
#		\frac{dZ_a}{dt} = -k_1*Z_a	                Z_a(0) = 1
#		\frac{dZ_b}{dt} = k_1*Z_a - k_2*Z_b		Z_b(0) = 0
#               \frac{dZ_c}{dt} = k_2*Z_b	                Z_c(0) = 0
#               C_k(t_i) = Z_k(t_i) + w(t_i)    for all t_i in measurement points
#               D_{i,j} = \sum_{k=0}^{Nc}C_k(t_i)S(l_j) + \xi_{i,j} for all t_i, for all l_j 
#       Initial concentration 

from __future__ import print_function
from kipet.library.TemplateBuilder import TemplateBuilder
from kipet.library.PyomoSimulator import PyomoSimulator
from kipet.library.ParameterEstimator import ParameterEstimator
from kipet.library.VarianceEstimator import VarianceEstimator
from kipet.library.data_tools import *
from kipet.library.MultipleExperimentsEstimator import MultipleExperimentsEstimator
import matplotlib.pyplot as plt
import os
import sys
import inspect
import six

if __name__ == "__main__":

    with_plots = True
    if len(sys.argv)==2:
        if int(sys.argv[1]):
            with_plots = False
 
    #=========================================================================
    #USER INPUT SECTION - REQUIRED MODEL BUILDING ACTIONS
    #=========================================================================
       
    # Load spectral data from the relevant file location. As described in section 4.3.1
    #################################################################################
    dataDirectory = os.path.abspath(
        os.path.join( os.path.dirname( os.path.abspath( inspect.getfile(
            inspect.currentframe() ) ) ), 'data_sets'))
    filename1 = os.path.join(dataDirectory,'Dij_multexp_tiv_G.txt')
    filename2 = os.path.join(dataDirectory,'Dij_multexp_tv_G.txt')
    filename3 = os.path.join(dataDirectory,'Dij_multexp_no_G.txt')
    D_frame1 = read_file(filename1)
    D_frame2 = read_file(filename2)
    D_frame3 = read_file(filename3)

    #This function can be used to remove a certain number of wavelengths from data
    # in this case only every 2nd wavelength is included
    # D_frame1 = decrease_wavelengths(D_frame1,A_set = 2)
    
    #Here we add noise to datasets in order to make our data differenct between experiments
    # D_frame2 = add_noise_to_signal(D_frame2, 0.00001)
    
    # D_frame2 = decrease_wavelengths(D_frame2,A_set = 2)

    #################################################################################    
    builder = TemplateBuilder()    
    components = {'A':0.01,'B':0,'C':0}
    builder.add_mixture_component(components)
    builder.add_parameter('k1', init=1.3, bounds=(0.0,2.0)) 
    #There is also the option of providing initial values: Just add init=... as additional argument as above.
    builder.add_parameter('k2', bounds=(0.0,0.5))
    
    builder.add_qr_bounds_init(bounds = (0,None),init = 1.0)
    builder.add_g_bounds_init(bounds = (0,None))
    
    # If you have multiple experiments, you need to add your experimental datasets to a dictionary:
    datasets = {'Exp1': D_frame1, 'Exp2': D_frame2, 'Exp3': D_frame3}
    # Additionally, we do not add the spectral data to the TemplateBuilder, rather supplying the 
    # TemplateBuilder before data is added as an argument into the function
    
    # define explicit system of ODEs
    def rule_odes(m,t):
        exprs = dict()
        exprs['A'] = -m.P['k1']*m.Z[t,'A']
        exprs['B'] = m.P['k1']*m.Z[t,'A']-m.P['k2']*m.Z[t,'B']
        exprs['C'] = m.P['k2']*m.Z[t,'B']
        return exprs
    
    builder.set_odes_rule(rule_odes)
    #opt_model = builder.create_pyomo_model(,10.0)
    start_time = {'Exp1':0.0, 'Exp2':0.0, 'Exp3':0.0}
    #, 'Exp3':0.0}
    end_time = {'Exp1':10.0, 'Exp2':10.0, 'Exp3':10.0}
    #, 'Exp3':10.0}
    
    options = dict()
    options['linear_solver'] = 'ma27'
    #options['mu_init']=1e-6
    
    # ============================================================================
    #   USER INPUT SECTION - MULTIPLE EXPERIMENTAL DATASETS       
    # ===========================================================================
    # Here we use the class for Multiple experiments, notice that we add the dictionary
    # Containing the datasets here as an argument
    pest = MultipleExperimentsEstimator(datasets)
    
    nfe = 100
    ncp = 3

    # Now we run the variance estimation on the problem. This is done differently to the
    # single experiment estimation as we now have to solve for variances in each dataset
    # separately these are automatically patched into the main model when parameter estimation is run
    results_variances = pest.run_variance_estimation(solver = 'ipopt', 
                                                      tee=True,
                                                      nfe=nfe,
                                                      ncp=ncp, 
                                                      solver_opts = options,
                                                      start_time=start_time, 
                                                      end_time=end_time, 
                                                      builder = builder)
    
    # Finally we run the parameter estimation. This solves each dataset separately first and then
    # links the models and solves it simultaneously
    
    Ex1_St = dict()
    Ex1_St["r1"] = [-1,1,0]
    Ex1_St["r2"] = [0,-1,0]
    
    # Input the information for each experiment in unanted_G_info as python dictionary.
    # In this exampls, only Exp1 and Exp2 have unwanted contributions while Exp3 don't have.
    unwanted_G_info = {"Exp1":{"type":"time_invariant_G","St":Ex1_St},
                       "Exp2":{"type":"time_variant_G"}}
    
    results_pest = pest.run_parameter_estimation(builder = builder,
                                                         tee=True,
                                                         nfe=nfe,
                                                         ncp=ncp,
                                                         # sigma_sq = variances,
                                                         solver_opts = options,
                                                         start_time=start_time, 
                                                         end_time=end_time,
                                                         unwanted_G_info = unwanted_G_info,
                                                         shared_spectra = True,
                                                         # Sometimes the estimation problem is easily to solve when the variances are scaled.
                                                         scaled_variance = True)                                                          
                                                         
    
    # Note here, that with the multiple datasets, we are returning a dictionary cotaining the 
    # results for each block. Since we know that all parameters are shared, we only need to print
    # the parameters from one experiment, however for the plots, they could differ between experiments
    print("The estimated parameters are:")
    #for k,v in six.iteritems(results_pest['Exp1'].P):
    #    print(k, v)
    for k,v in results_pest.items():
        print(results_pest[k].P)
            
    if with_plots:
        for k,v in results_pest.items():

            results_pest[k].Z.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
            
            results_pest[k].C.plot.line(legend=True)
            plt.xlabel("time (s)")
            plt.ylabel("Concentration (mol/L)")
            plt.title("Concentration Profile")
            
            results_pest[k].S.plot.line(legend=True)
            plt.xlabel("Wavelength (cm)")
            plt.ylabel("Absorbance (L/(mol cm))")
            plt.title("Absorbance  Profile")
        
            plt.show()
            
