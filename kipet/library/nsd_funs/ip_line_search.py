import numpy as np
from scipy.optimize import Bounds
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm

__all__ = ['ip_line_search']

class ScalarFunction(object):
    """Scalar function and its derivatives.

    This class defines a scalar function F: R^n->R and its first and 
    second derivatives.

    """
    def __init__(self, fun, x0, args, grad, hess, mu):

        if not callable(fun):
            raise ValueError("`fun` must be callable")

        if not callable(grad):
            raise ValueError("`grad` must be callable")
        
        if not callable(hess):
            raise ValueError("`hess` must be callable")

        self.x = x0
        self.mu = mu
        self.n = self.x.size
        self.nfev = 0
        self.ngev = 0
        self.nhev = 0
        self.f_updated = False
        self.g_updated = False
        self.H_updated = False

        # Function evaluation
        def fun_wrapped(x, mu):
            self.nfev += 1
            return fun(x, mu, *args)
        
        def update_fun():
            self.f = fun_wrapped(self.x, self.mu)
        
        self._update_fun_impl = update_fun
        self._update_fun()


        # Gradient evaluation
        def grad_wrapped(x, mu):
            self.ngev += 1
            return grad(x, mu, *args)
        
        def update_grad():
            self.g = grad_wrapped(self.x, self.mu)

        self._update_grad_impl = update_grad
        self._update_grad()

        
        # Hessian evaluation
        def hess_wrapped(x, mu):
            self.nhev += 1
            return hess(x, mu, *args)

        def update_hess():
            self.H = hess_wrapped(self.x, self.mu)

        self._update_hess_impl = update_hess
        self._update_hess()

        # Update x
        def update_x(x):
            self.x = x
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
        
        self._update_x_impl = update_x

        # Update mu
        def update_mu(mu):
            self.mu = mu
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
        
        self._update_mu_impl = update_mu

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_grad(self):
        if not self.g_updated:
            self._update_grad_impl()
            self.g_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True
    # wrap functions to evaluete only if x is updated
    def fun(self, x, mu):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        if self.mu != mu:
            self._update_mu_impl(mu)
        self._update_fun()
        return self.f

    def grad(self, x, mu):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        if self.mu != mu:
            self._update_mu_impl(mu)
        self._update_fun() # This is for the NSD which needs to update the model. fun update the model.
        self._update_grad()
        return self.g

    def hess(self, x, mu):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)
        if self.mu != mu:
            self._update_mu_impl(mu)
        self._update_fun() # This is for the NSD which needs to update the model. fun update the model.
        self._update_hess()
        return self.H

class OptimizationResult(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

class ReportBase(object):
    COLUMN_NAMES = NotImplemented
    COLUMN_WIDTHS = NotImplemented
    ITERATION_FORMATS = NotImplemented

    @classmethod
    def print_header(cls):
        fmt = ("|"
               + "|".join(["{{:^{}}}".format(x) for x in cls.COLUMN_WIDTHS])
               + "|")
        separators = ['-' * x for x in cls.COLUMN_WIDTHS]
        print(fmt.format(*cls.COLUMN_NAMES))
        print(fmt.format(*separators))

    @classmethod
    def print_iteration(cls, *args):
        # args[3] is obj func. It should really be a float. However,
        # trust-constr typically provides a length 1 array. We have to coerce
        # it to a float, otherwise the string format doesn't work.
        args = list(args)
        args[3] = float(args[3])

        iteration_format = ["{{:{}}}".format(x) for x in cls.ITERATION_FORMATS]
        fmt = "|" + "|".join(iteration_format) + "|"
        print(fmt.format(*args))

    @classmethod
    def print_footer(cls):
        print()


class BasicReport(ReportBase):
    COLUMN_NAMES = ["niter", "obj", "mu", "||d||", "alpha_x",
                    "alpha_dual", "nl", "error", "dual_error", "comp_error"]
    COLUMN_WIDTHS = [7, 13, 10, 10, 10, 10, 7, 13, 13, 13]
    ITERATION_FORMATS = ["^7", "^+13.4e", "^10.2e", "^10.2e", "^10.2e", "^10.2e",
                            "^7", "^+13.4e", "^+13.4e", "^+13.4e"]

class BarrierSubproblem:
    """
    Barrier problem:
        min f(x) - mu*sum(log(x -x.lb)) - mu*sum(log(x.ub - x))

    """

    def __init__(self, fun, x0, grad, hess, bounds, mu, tau_min, kappa_s):
        self.fun = fun
        self.x = x0
        self.grad = grad
        self.hess = hess
        self.mu = mu
        self.tau_min = tau_min
        self.tau = max(self.tau_min, 1- self.mu)
        self.kappa_s = kappa_s
        self.bounds = bounds
        self.s_lb, self.s_ub = self.get_slack(self.x)
        self.dual_U = self.mu/self.s_ub
        self.dual_L = self.mu/self.s_lb
        self.fk = self.get_function(self.x)
        self.gk = self.get_gradient(self.x)
        self.Hk = self.get_hessian(self.x)
        self.dual_error = self.get_dual_infeasibility()
        self.comp_L_error, self.comp_U_error = self.get_comp_infeasibility()
        self.error = max(self.dual_error, self.comp_L_error, self.comp_U_error)


    def update(self, x, da_U, da_L, alpha_a):
        self.update_x(x)
        self.s_lb, self.s_ub = self.get_slack(self.x)
        self.update_dual(da_U, da_L, alpha_a)
        self.fk = self.get_function(self.x)
        self.gk = self.get_gradient(self.x)
        self.Hk = self.get_hessian(self.x)
        self.dual_error = self.get_dual_infeasibility()
        self.comp_L_error, self.comp_U_error = self.get_comp_infeasibility()
        self.error = max(self.dual_error, self.comp_L_error, self.comp_U_error) 

    def update_mu(self, mu):
        self.mu = mu
        self.tau = max(self.tau_min, 1 - self.mu)
    
    def update_x(self, x):
        self.x = x
    
    def update_dual(self, da_U, da_L, alpha_a):
        dual_U = self.dual_U + alpha_a*da_U
        dual_L = self.dual_L + alpha_a*da_L
        dual_L = np.maximum(np.minimum(dual_L, self.kappa_s*self.mu/(self.x - self.bounds.lb)), self.mu/(self.kappa_s*(self.x - self.bounds.lb)))
        dual_U = np.maximum(np.minimum(dual_U, self.kappa_s*self.mu/(self.bounds.ub - self.x)), self.mu/(self.kappa_s*(self.bounds.ub - self.x)))
        self.dual_U = dual_U
        self.dual_L = dual_L

    def get_slack(self, x):
        
        s_lb = x - self.bounds.lb
        s_ub = self.bounds.ub - x
        return s_lb, s_ub

    def get_function(self, x):
        """Return the barrier function at given point.
        """
        # Get slacl variables
        s_lb, s_ub = self.get_slack(x)
        # Compute barrier function
        f = self.fun(x, self.mu)
        
        log_s_lb =[np.log(s_lb_i) if s_lb_i >0 else -np.inf for s_lb_i in s_lb]
        log_s_ub =[np.log(s_ub_i) if s_ub_i >0 else -np.inf for s_ub_i in s_ub]
        
        return f - self.mu*np.sum(log_s_lb) - self.mu*np.sum(log_s_ub)

    def get_gradient(self, x):
        """Update the gradient of the barrier function
        """
        # # Get slack variables
        # s_lb, s_ub = self.get_slack(x)
        # Compute the gradient of the objective
        g = self.grad(x, self.mu)
        
        return g - self.mu/self.s_lb + self.mu/self.s_ub

    def get_hessian(self, x):
        """Update the hessian of the barrier function
        """
        # # Get slack variables
        # s_lb, s_ub = self.get_slack(x)
        # Compute the Hessian of the objective
        H = self.hess(x, self.mu)
        H_aug_diag = self.dual_L/self.s_lb + self.dual_U/self.s_ub
        H_aug = csc_matrix(np.diag(H_aug_diag)) 
        return H + H_aug

    def get_dual_infeasibility(self):
        
        g = self.grad(self.x, self.mu)
        dual_error = g - self.dual_L + self.dual_U
        return norm(dual_error, np.inf)

    def get_comp_infeasibility(self):

        comp_L_error = self.dual_L*self.s_lb - self.mu
        comp_U_error = self.dual_U*self.s_ub - self.mu
        return norm(comp_L_error, np.inf), norm(comp_U_error, np.inf)

    def dual_step(self, dx):
        """Evaluete the Newton step for duals
        """
        D_L_inv = 1.0/(self.x - self.bounds.lb)
        D_U_inv = 1.0/(self.bounds.ub - self.x)
        da_U = -self.dual_U + self.mu*D_U_inv + D_U_inv*self.dual_U*dx
        da_L = -self.dual_L + self.mu*D_L_inv - D_L_inv*self.dual_L*dx
        return da_U, da_L

    def frac_bound_x(self, dx):
        """Evaluate the alpha_k^{max} which satisfies "fraction to the bound" rule
        """
       
        alpha_L = self.tau*(self.bounds.lb - self.x)/dx
        alpha_U = self.tau*(self.bounds.ub - self.x)/dx

        alpha_max = min(np.maximum(alpha_L, alpha_U))

        if alpha_max > 1.0:
            alpha_max = 1.0
        elif alpha_max < 0.0:
            raise ValueError("alpha_max is negative")
        return alpha_max

    def get_alpha_dual(self, da_U, da_L):
        """Evaluate alpha for duals
        """
        if (all(da_U > 0) and all(da_L > 0)):
            alpha_a = 1.0
        else:
            alpha_U = self.tau*(- self.dual_U)/da_U
            alpha_L = self.tau*(- self.dual_L)/da_L
            alpha_a = min([v for v in np.concatenate([alpha_U, alpha_L]) if v >=0])

        if alpha_a > 1.0:
            alpha_a = 1.0
        elif alpha_a < 0.0:
            raise ValueError("alpha_a is negative")
        return alpha_a


    def global_stop_criteria(self, state, e_tol, max_iters, callback):

        if state.iters == 0:
            BasicReport.print_header()
        
        BasicReport.print_iteration(state.iters, state.obj, state.mu, state.d_norm,
                                    state.alpha_x, state.alpha_a, state.l, state.error,
                                    state.dual_error, state.comp_error )

        if callback is not None and callback(np.copy(state.x), state):
            state.stop_status = 4
            return True
            
        elif (self.error <= e_tol):
            state.stop_status = 1
            return True
        
        elif (state.iters >= max_iters):
            state.stop_status = 2
            return True

        elif state.step_cond and (self.mu < e_tol/10):
            state.stop_status = 3
            return True
        
    def mu_update_criteria(self, state, kappa_e):

        if (state.step_cond) or ((self.error < kappa_e * self.mu) and (state.iters > 0)):
            return True

def bound_push(x, x_U, x_L, kappa_1 = 1.0e-2, kappa_2 = 1.0e-2):
    """
    This provides the bound_push for the initial value of x

    Args:
        x (np.array): The initial value of x.

        x_U (np.array): The upper bounds of x.

        x_L (np.array): The lower bounds of x.

        kappa_p : the bound push parameter.

    Return:
        x (np.array): The pushed x.
    """
    pL = np.minimum(kappa_1*np.maximum(1, abs(x_L)), kappa_2*(x_U - x_L))
    pU = np.minimum(kappa_1*np.maximum(1, abs(x_U)), kappa_2*(x_U - x_L))
    x_L_p = x_L + pL
    x_U_p = x_U - pU

    for i, v in enumerate(x):
        if v < x_L_p[i]:
            x[i] = x_L_p[i]
        elif v > x_U_p[i]:
            x[i] = x_U_p[i]
        else:
            continue
    return x

def ip_line_search(fun, x0, args=(), grad=None, hess=None, 
                   bounds=None, callback =None, 
                   e_tol=1e-8, tau_min=0.99,
                   mu_init = 0.1, kappa_e = 1.0, 
                   kappa_mu = 0.2, theta_mu = 1.5, 
                   eta = 1e-4, e_mach = 1.0e-16, 
                   kappa_1 = 1e-2, kappa_2 = 1e-2, 
                   kappa_s = 1e+10, max_iters = 100):

    """ Minimization of scalar function of one or more variables with
        bounds

    Args:
        fun : callable
            The objective function to be minimized.

                ``fun(x, mu, *args) -> float``
            
            where ``x`` is an 1-D array with shape (n, ) and ``args``
            is a tuple of the fixed parameters needed to completely
            specify the function.
        x0  : ndarray, shape (n, )
            Initial guess. 
        args  : tuple
            Extra arguments passed to the objective function and its
            derivatives (`fun`, `grad` and `hess` functions)
        grad : callable
            The gradient vector of the objective function.
            It should be a function that returns the gradient vector:

                ``grad(x, *args) -> array_like, shape (n,)``

            where ``x`` is an array with shape (n, ) and ``args`` is a tuple with 
            the fixed parameters.
        hess : callable
            The hessian matrix of the objective function.
            It should be a function that returns the hessian matrix:

                ``hess(x, *args) -> csc_matrix , (n, n)

            where ``x`` is an array with shape (n, ) and ``args`` is a tuple with 
            the fixed parameters.
        bounds : sequence of `Bounds` class.
            1. Instance of `Bounds` class. (scipy.optimize.Bounds)
            2. Sequence of ``(min, max)`` pairs for each element in `x`.
            all variables are expected to have the bounds.
        callback  : callable
            Called after each iteration.

                ``callback(xk, state) -> bool``
            
            where ``xk`` is the current parameter vector. and ``state`` is
            an optimization result object. If callback returns True, the algorithm
            execution is terminated.
        
    Returns:
        res : Optimization result

    """
    # Make x0 as ndarray
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float) # Make x0 values float, if it is integer.

    # if args not tuple, convert it to tuple. 
    if not isinstance(args, tuple):
        args = (args,)

    # if bounds is not Bounds object, it is converted.
    if not isinstance(bounds, Bounds):
        lb, ub = zip(*bounds)
        lb = np.array(lb, dtype=float)
        ub = np.array(ub, dtype=float)
        bounds = Bounds(lb, ub)

    # Perturb the initial x with bound push
    bound_push(x0, bounds.ub, bounds.lb, kappa_1, kappa_2)
    
    # Set the objective function and its derivatives
    objective = ScalarFunction(fun, x0, args, grad, hess, mu_init)
    
    # Construnct initialized the barrier problem
    bp = BarrierSubproblem(objective.fun, x0, objective.grad, objective.hess, 
                            bounds, mu_init, tau_min, kappa_s)

    # Checker for Step size condition
    step_cond = [False, False]

    # Initialize iteration counter
    iters = 0

    # Get initial state
    state = OptimizationResult(iters = iters, nfev = objective.nfev, ngev = objective.ngev,
                               nhev = objective.nhev, obj = bp.fk, x = bp.x, mu = bp.mu,
                               grad = bp.gk, hess =bp.Hk, dx = np.zeros(bp.x.shape), d_norm = 0,
                               daU = np.zeros(bp.x.shape), daL = np.zeros(bp.x.shape), 
                               alpha_x = 0, alpha_a = 0, aU = bp.dual_U, aL = bp.dual_L, 
                               l = 0, error = bp.error, dual_error = bp.dual_error, 
                               comp_error = max(bp.comp_L_error, bp.comp_U_error),
                               step_cond = all(step_cond), stop_status= 0 )

    while not bp.global_stop_criteria(state, e_tol, max_iters, callback):

        if bp.mu_update_criteria(state, kappa_e):
            mu_new = max(e_tol/10, min(kappa_mu*bp.mu, bp.mu**theta_mu))
            bp.update_mu(mu_new)          

        # get objective, gradient, and Hessian of x_k
        x_k = bp.x
        f_k = bp.get_function(bp.x)
        g_k = bp.get_gradient(bp.x)
        H_k = bp.get_hessian(bp.x)

        # get Newton step dx and da
        
        dx = spsolve(H_k, -g_k)
        daU, daL = bp.dual_step(dx) 

        ## Line search
        # initialization
        alpha_x_max = bp.frac_bound_x(dx)
        alpha_x = alpha_x_max
        iters_l = 0 

        while True:
            iters_l += 1
            
            print(f'Inside the step length calc: {iters_l}')
            
            x_new = x_k + alpha_x*dx
            f_new = bp.get_function(x_new)

            print(f'f_new in the while: {f_new}')
            print(f'{f_new} <= {f_k} + {eta*alpha_x*np.dot(g_k, dx)}')
            ## Check Armijo condition
            if (f_new <= f_k + eta*alpha_x*np.dot(g_k, dx)):
                print('Armijo passed')
                break
            else:
                print('Armijo failed')
                
                print("Interpolating")
                print(f'alpha_before: {alpha_x}')
                # Quadratic interpolation for new alpha_x
                alpha_x = 0.5*g_k.dot(dx)*alpha_x**2/(f_k - f_new + g_k.dot(dx)*alpha_x)
                if isinstance(alpha_x, np.matrix):
                    alpha_x = alpha_x.item()
                
                # YOU NEED AN OUT IF ALPHA IS TOO SMALL HERE
                if iters_l > 20:
                    break
                
                
            print(f'alpha_after: {alpha_x}')
                
        ## Step size check
        step_cond.pop(0) 
        step_status = False
        
        print(f'the alpha min is: {alpha_x_max} ########################################')
        
        if ((not (alpha_x < alpha_x_max)) and (max(abs(dx)/(1 + abs(x_k))) < 10*e_mach)):
            step_status = True
        step_cond.append(step_status)

        ## Evaluate alpha_a for duals
        alpha_a = bp.get_alpha_dual(daU, daL)

        ## Update the barrier subproblem
        bp.update(x_new, daU, daL, alpha_a)
        iters += 1
        ## Update state
        state = OptimizationResult(iters = iters, nfev = objective.nfev, ngev = objective.ngev,
                                nhev = objective.nhev, obj = bp.fk, x = bp.x, mu = bp.mu,
                                grad = bp.gk, hess =bp.Hk, dx = dx, d_norm = norm(dx, np.inf), daU = daU, daL = daL, 
                                alpha_x = alpha_x, alpha_a = alpha_a, aU = bp.dual_U, aL = bp.dual_L, 
                                l = iters_l, error = bp.error, dual_error = bp.dual_error, 
                                comp_error = max(bp.comp_L_error, bp.comp_U_error),
                                step_cond = all(step_cond), stop_status= 0 )


    # Termination message
    if state.stop_status == 1:
        print("EXIT: OPTIMAL SOLUTION FOUND")
    elif state.stop_status == 2:
        print("EXIT: MAXIMUM NO. OF ITERATIONS REACHED")
    elif state.stop_status == 3:
        print("EXIT: SEARCH DIRECTION TOO SMALL")

    return state