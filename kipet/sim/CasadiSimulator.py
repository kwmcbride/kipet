import casadi as ca
from casadi.tools import *
from ResultsObject import *
from Simulator import *
import copy


class CasadiSimulator(Simulator):
    def __init__(self,model):
        super(CasadiSimulator, self).__init__(model)
        self.nfe = None
        self._times = set([t for t in model.meas_times])
        self._n_times = len(self._times)
        self._spectra_given = hasattr(self.model, 'D')
        
    def apply_discretization(self,transformation,**kwargs):
        
        if kwargs.has_key('nfe'):
            self.nfe = kwargs['nfe']
            self.model.start_time
            step = (self.model.end_time - self.model.start_time)/self.nfe
            for i in xrange(0,self.nfe+1):
                self._times.add(i*step)
                
            self._n_times = len(self._times)
            self._discretized = True
        else:
            raise RuntimeError('Specify discretization points nfe=int8')
        
        
    def initialize_from_trajectory(self,trajectory_dictionary):
        pass

    def run_sim(self,solver,tee=False,solver_opts={},sigmas=None,seed=None):

        # adjusts the seed to reproduce results with noise
        np.random.seed(seed)
        
        Z_var = self.model.Z
        X_var = self.model.X
        
        if self._discretized is False:
            raise RuntimeError('apply discretization first before runing simulation')
        states_l = []
        ode_l = []
        init_conditions_l = []


        for i,k in enumerate(self._mixture_components):
            states_l.append(Z_var[k])

            expr = self.model.odes[k]
            
            if isinstance(expr,ca.SX):
                representation = expr.getRepresentation()
            else:
                representation = str(expr)
            if 'nan' not in representation:
                ode_l.append(expr)
            else:
                raise RuntimeError('Mass balance expression for {} is nan.\n'.format(k)+
                'This usually happens when not using casadi.operator\n'+ 
                'e.g casadi.exp(expression)\n')
            init_conditions_l.append(self.model.init_conditions[k])

        for i,k in enumerate(self._complementary_states):
            states_l.append(X_var[k])
            expr = self.model.odes[k]
            if isinstance(expr,ca.SX):
                representation = expr.getRepresentation()
            else:
                representation = str(expr)
            if 'nan' not in representation:
                ode_l.append(expr)
            else:
                raise RuntimeError('Complementary ode expression for {} is nan.\n'.format(k)+
                'This usually happens when not using casadi.operator\n'+ 
                'e.g casadi.exp(expression)')
            init_conditions_l.append(self.model.init_conditions[k])

        states = ca.vertcat(*states_l)
        ode = ca.vertcat(*ode_l)
        x_0 = ca.vertcat(*init_conditions_l)

        system = {'x':states, 'ode':ode}
        
        step = (self.model.end_time - self.model.start_time)/self.nfe

        results = ResultsObject()

        fun_ode = ca.Function("odeFunc",[states],[ode])

        c_results =  []
        dc_results = []

        x_results =  []
        dx_results = []

        xk = x_0
        times = sorted(self._times)
        for i,t in enumerate(times):
            if t == self.model.start_time:
                odek = fun_ode(xk)
                for j,w in enumerate(init_conditions_l):
                    if j<self._n_components:
                        c_results.append(w)
                        dc_results.append(odek[j])
                    else:
                        x_results.append(w)
                        dx_results.append(odek[j])
            else:
                step = t - times[i-1]
                opts = {'tf':step,'print_stats':tee,'verbose':False}
                I = integrator("I",solver, system, opts)
                xk = I(x0=xk)['xf']

                # check for nan
                for j in xrange(xk.numel()):
                    if np.isnan(float(xk[j])):
                        raise RuntimeError('The iterator returned nan. exiting the program')
                    
                odek = fun_ode(xk)
                
                for j,k in enumerate(self._mixture_components):
                    c_results.append(xk[j])
                    dc_results.append(odek[j])

                for i,k in enumerate(self._complementary_states):
                    j = i+self._n_components
                    x_results.append(xk[j])
                    dx_results.append(odek[j])
        
        c_array = np.array(c_results).reshape((self._n_times,self._n_components))
        results.Z = pd.DataFrame(data=c_array,columns=self._mixture_components,index=times)
                    
        dc_array = np.array(dc_results).reshape((self._n_times,self._n_components))
        results.dZdt = pd.DataFrame(data=dc_array,columns=self._mixture_components,index=times)

        x_array = np.array(x_results).reshape((self._n_times,self._n_complementary_states))
        results.X = pd.DataFrame(data=x_array,columns=self._complementary_states,index=times)

        dx_array = np.array(dx_results).reshape((self._n_times,self._n_complementary_states))
        results.dXdt = pd.DataFrame(data=dx_array,columns=self._complementary_states,index=times)
        
        
        if self._spectra_given:
            # solves over determined system
            D_data = self.model.D
            c_noise_array, s_array = self._solve_CS_from_D(results.Z,tee=tee)

            d_results = []
            for t in self._meas_times:
                for l in self._meas_lambdas:
                    d_results.append(D_data[t,l])
                    
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
        else:

            w = np.zeros((self._n_components,self._n_meas_times))
            # for the noise term
            if sigmas:
                for i,k in enumerate(self._mixture_components):
                    if sigmas.has_key(k):
                        sigma = sigmas[k]**0.5
                        dw_k = np.random.normal(0.0,sigma,self._n_meas_times)
                        w[i,:] = np.cumsum(dw_k)
            
            c_noise_results = []
            for i,t in enumerate(self._meas_times):
                for j,k in enumerate(self._mixture_components):
                    c_noise_results.append(results.Z[k][t]+w[j,i])

            s_results = []
            for l in self._meas_lambdas:
                for k in self._mixture_components:
                    s_results.append(self.model.S[l,k])

            d_results = []
            if sigmas:
                sigma_d = sigmas.get('device')**0.5 if sigmas.has_key('device') else 0
            else:
                sigma_d = 0
            if s_results and c_noise_results:
                for i,t in enumerate(self._meas_times):
                    for j,l in enumerate(self._meas_lambdas):
                        suma = 0.0
                        for w,k in enumerate(self._mixture_components):
                            C = c_noise_results[i*self._n_components+w]
                            S = s_results[j*self._n_components+w]
                            suma+= C*S
                        if sigma_d:
                            suma+= np.random.normal(0.0,sigma_d)
                        d_results.append(suma)

            c_noise_array = np.array(c_noise_results).reshape((self._n_meas_times,self._n_components))
            s_array = np.array(s_results).reshape((self._n_meas_lambdas,self._n_components))
            d_array = np.array(d_results).reshape((self._n_meas_times,self._n_meas_lambdas))
            
        # stores everything in restuls object
        results.C = pd.DataFrame(data=c_noise_array,
                                       columns=self._mixture_components,
                                       index=self._meas_times)
        results.S = pd.DataFrame(data=s_array,
                                 columns=self._mixture_components,
                                 index=self._meas_lambdas)
        results.D = pd.DataFrame(data=d_array,
                                 columns=self._meas_lambdas,
                                 index=self._meas_times)
        
        param_vals = dict()
        for name in self.model.parameter_names:
            param_vals[name] = self.model.P[name]

        results.P = param_vals
        
        return results
        
        
        
