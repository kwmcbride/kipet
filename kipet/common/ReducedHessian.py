"""
Reduced Hessian Generation

This module creates the reduced Hessian for use in various KIPET modules
"""
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.environ import (
    Constraint,
    Param,
    Set,
    SolverFactory,
    Suffix,
    )
from scipy.sparse import coo_matrix, triu
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
    
from kipet.common.parameter_handling import (
    set_scaled_parameter_bounds,
    )
 
DEBUG = False

class ReducedHessian(object):
    """Class for handling the reduced hessian calculations in KIPET/NSD"""
    
    def __init__(self, 
                 model_object,
                 kkt_method = 'k_aug',
                 global_param_name = 'd',
                 param_con_method = 'global',
                 scaled = False,
                 rho = 10,
                 set_param_bounds = False,
                 parameter_set = None,
                 variable_name = 'P',
                 param_set_name = 'parameter_names',
                 current_set = 'current_p_set',
                 set_up_constraints = True,
                 use_duals = True,
                 global_constraint_name = 'fix_params_to_global',
                 file_number = None
                 ):
        
        self.model_object =  model_object
        self.kkt_method = kkt_method
        
        self.global_param_name = global_param_name
        self.global_constraint_name = global_constraint_name
        self.param_con_method = param_con_method
        self.scaled = scaled
        self.rho = rho
        self.set_param_bounds = set_param_bounds
        self.parameter_set = parameter_set
        self.variable_name = variable_name
        self.param_set_name = param_set_name
        self.current_set = current_set
        self.set_up_constraints = set_up_constraints
        self.use_duals = use_duals
        self.file_number = file_number
        
        self.verbose = DEBUG


    def get_tmp_file(self):
        
        file_tag = ''
        if self.file_number is not None:
            file_tag += f'_{self.file_number}'
            
        return 'ipopt_output' + file_tag

    def get_file_info(self):
        
        tmpfile_i = self.get_tmp_file()
            
        with open(tmpfile_i, 'r') as f:
            output_string = f.read()
        
        stub = output_string.split('\n')[0].split(',')[1][2:-4]
        
        nl_file = Path(stub + '.nl')
        col_file = Path(stub + '.col')
        row_file = Path(stub + '.row')
        sol_file = Path(stub + '.sol')
        
        self.sol_files = dict(
            nl = nl_file,
            col = col_file,
            row = row_file,
            sol = sol_file,
            )
        
        return None
    
    def delete_sol_files(self):
        
        if hasattr(self, 'sol_files'):
        
            for key, file in self.sol_files.items():
                file.unlink()
                
            del self.sol_files
            
        return None

    def get_kkt_info(self):
        
        """Takes the model and uses PyNumero to get the jacobian and Hessian
        information as dataframes
        
        Args:
            model_object (pyomo ConcreteModel): A pyomo model instance of the current
                problem (used in calculating the reduced Hessian)
    
            method (str): defaults to k_aug, method by which to obtain optimization
                results
    
        Returns:
            
            kkt_data (dict): dictionary with the following structure:
                
                    {
                    'J': J,   # Jacobian
                    'H': H,   # Hessian
                    'var_ind': var_index_names, # Variable index
                    'con_ind': con_index_names, # Constraint index
                    'duals': duals, # Duals
                    }
            
        """
        self.get_file_info()
        
        if self.kkt_method == 'pynumero':
        
            nlp = PyomoNLP(self.model_object)
            varList = nlp.get_pyomo_variables()
            conList = nlp.get_pyomo_constraints()
            duals = nlp.get_duals()
            
            J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
            H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
            J = csc_matrix(J)
            
            var_index_names = [v.name for v in varList]
            con_index_names = [v.name for v in conList]
            
        elif self.kkt_method == 'k_aug':
        
            kaug = SolverFactory('k_aug')
            
            kaug.options["deb_kkt"] = ""  
            kaug.solve(self.model_object, tee=False)
            
            hess = pd.read_csv('hess_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
            hess.columns = ['irow', 'jcol', 'vals']
            hess.irow -= 1
            hess.jcol -= 1
            os.unlink('hess_debug.in')
            
            jac = pd.read_csv('jacobi_debug.in', delim_whitespace=True, header=None, skipinitialspace=True)
            m = jac.iloc[0,0]
            n = jac.iloc[0,1]
            jac.drop(index=[0], inplace=True)
            jac.columns = ['irow', 'jcol', 'vals']
            jac.irow -= 1
            jac.jcol -= 1
            os.unlink('jacobi_debug.in')
            
            #try:
            #    duals = read_duals(stub + '.sol')
            #except:
            duals = None
            
            J = coo_matrix((jac.vals, (jac.irow, jac.jcol)), shape=(m, n)) 
            Hess_coo = coo_matrix((hess.vals, (hess.irow, hess.jcol)), shape=(n, n)) 
            H = Hess_coo + triu(Hess_coo, 1).T
            
            var_index_names = pd.read_csv(self.sol_files['col'], sep = ';', header=None) # dummy sep
            con_index_names = pd.read_csv(self.sol_files['row'], sep = ';', header=None) # dummy sep
            
            var_index_names = [var_name for var_name in var_index_names[0]]
            con_index_names = [con_name for con_name in con_index_names[0].iloc[:-1]]
            con_index_number = {v: k for k, v in enumerate(con_index_names)}
        
        self.delete_sol_files()
        
        self.kkt_data = {
                    'J': J,
                    'H': H,
                    'var_ind': var_index_names,
                    'con_ind': con_index_names,
                    'duals': duals,
                    }
        
        return None
    
    def get_kkt_df(self):
        
        self.get_kkt_info()
        
        H = self.kkt_data['H']
        var_index_names = self.kkt_data['var_ind']
        con_index_names = self.kkt_data['con_ind']
        
        H_df = pd.DataFrame(H.todense(), columns=var_index_names, index=var_index_names)
    
        # See if this works
    
        # h_con_indx = [k for k in self.model.C.keys()]
        # h_con = [f'Z[{h[0]},{h[1]}]' for h in h_con_indx if h[1] in self.model.measured_data]
        # h_con_indx = [k for k in self.model.U.keys()]
        # h_con += [f'X[{h[0]},{h[1]}]' for h in h_con_indx  if h[1] in self.model.measured_data]
        
        
        # col_ind  = [var_ind.loc[var_ind[0] == v].index[0] for v in h_con]
        # Zr = Z[col_ind, :]
        
        # df_Zr = pd.DataFrame(Zr, index=h_con, columns=[k for k, v in self.model.P.items()])
        # df_Hv = H_df.loc[h_con, h_con]    
    
    
        return H_df
    
    
    
    def prep_model_for_k_aug(self):
        """This function prepares the optimization models with required
        suffixes.
        
        Args:
            model_object (pyomo model): The model of the system
            
        Retuns:
            None
            
        """
        self.model_object.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        # self.model_object.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        # self.model_object.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # self.model_object.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        # self.model_object.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
        # self.model_object.red_hessian = Suffix(direction=Suffix.EXPORT)
        # self.model_object.dof_v = Suffix(direction=Suffix.EXPORT)
        # self.model_object.rh_name = Suffix(direction=Suffix.IMPORT)
        
        count_vars = 1
        for k, v in self.model_object.P.items():
            self.model_object.dof_v[k] = count_vars
            count_vars += 1
        
        self.model_object.npdp = Suffix(direction=Suffix.EXPORT)
        
        return None
    
    def calculate_duals(self):
        """Get duals"""
    
        # For testing - very slow and should not be used!
        if self.kkt_method == 'pynumero':
        
            nlp = PyomoNLP(self.model_object)
            varList = nlp.get_pyomo_variables()
            conList = nlp.get_pyomo_constraints()
            duals = nlp.get_duals()
            
            J = nlp.extract_submatrix_jacobian(pyomo_variables=varList, pyomo_constraints=conList)
            H = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=varList, pyomo_variables_cols=varList)
            J = csc_matrix(J)
            
            var_index_names = [v.name for v in varList]
            con_index_names = [v.name for v in conList]                       
            
            dummy_constraints = [f'{self.global_constraint_name}[{k}]' for k in self.parameter_set]
            jac_row_ind = [con_index_names.index(d) for d in dummy_constraints] 
            duals_imp = [duals[i] for i in jac_row_ind]
        
            self.duals = dict(zip(self.parameter_set, duals_imp))
            if self.verbose:
                print(f'The pynumero results are:')
                print(self.duals)
            
        else:
                         
            self.duals = {key: self.model_object.dual[getattr(self.model_object, self.global_constraint_name)[key]] for key, val in getattr(self.model_object, self.global_param_name).items()}
        
            if self.verbose:
                print('The duals are:')
                print(self.duals)
        
        self.delete_sol_files()
        
        return self.duals              
    
    def optimize_model(self, d=None):
        """Takes the model object and performs the optimization
        
        Args:
            model_object (pyomo model): the pyomo model of the reaction
            
            parameter_set (list): list of current model parameters
            
        Returns:
            reduced_hessian (numpy array): reduced hessian of the model
        
        """
        if self.verbose:    
            print(f'd: {d}')
        
        ipopt = SolverFactory('ipopt')
        tmpfile_i = self.get_tmp_file()
    
        if self.param_con_method == 'global':
            
            if d is not None:
                param_val_dict = {p: d[i] for i, p in enumerate(self.parameter_set)}
                for k, v in getattr(self.model_object, self.variable_name).items():
                    v.set_value(param_val_dict[k])
        
            self.add_global_constraints() 
        
        elif self.param_con_method == 'fixed':
            
            if hasattr(self.model_object, self.global_constraint_name):
                self.model_object.del_component(self.global_constraint_name)  
        
            delta = 1e-20  
            for k, v in getattr(self.model_object, self.variable_name).items():
                if k in self.parameter_set:
                    ub = v.value
                    lb = v.value - delta
                    v.setlb(lb)
                    v.setub(ub)
                    v.unfix()
                else:
                    v.fix()
                    
        if self.set_param_bounds:
            set_scaled_parameter_bounds(self.model_object, 
                                        parameter_set=self.parameter_set, 
                                        rho=self.rho, 
                                        scaled=self.scaled)  
                    
        ipopt.solve(self.model_object, 
                    symbolic_solver_labels=True, 
                    keepfiles=True, 
                    tee=False,
                    logfile=tmpfile_i,
                    )
        
        # Create the file object so that it can be deleted
        self.get_file_info()
        
        return None

    def calculate_reduced_hessian(self, d=None, optimize=False, return_Z=False):
        """Calculate the reduced Hessian
        
        Args:
            model_object (pyomo model): the pyomo model of the reaction
            
            parameter_set (list): list of current model parameters
            
        Returns:
            reduced_hessian (numpy array): reduced hessian of the model
        
        """
        if optimize:
            self.optimize_model(d)
        
        self.get_kkt_info()
        H = self.kkt_data['H']
        J = self.kkt_data['J']
        var_ind = self.kkt_data['var_ind']
        con_ind_new = self.kkt_data['con_ind']
        duals = self.kkt_data['duals']
    
        col_ind = [var_ind.index(f'{self.variable_name}[{v}]') for v in self.parameter_set]
        m, n = J.shape  
     
        if self.param_con_method == 'global':
            
            dummy_constraints = [f'{self.global_constraint_name}[{k}]' for k in self.parameter_set]
            jac_row_ind = [con_ind_new.index(d) for d in dummy_constraints] 
            #duals_imp = [duals[i] for i in jac_row_ind]
            
            #print(J.shape, len(duals_imp))
    
            J_c = delete_from_csr(J.tocsr(), row_indices=jac_row_ind).tocsc()
            row_indexer = SparseRowIndexer(J_c)
            J_f = row_indexer[col_ind]
            J_f = delete_from_csr(J_f.tocsr(), row_indices=jac_row_ind, col_indices=[])
            J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)  
    
        elif self.param_con_method == 'fixed':
            
            jac_row_ind = []
            duals_imp = None
        
            J_c = J.tocsc()# delete_from_csr(J.tocsr(), row_indices=jac_row_ind).tocsc()
            row_indexer = SparseRowIndexer(J_c.T)
            J_f = row_indexer[col_ind].T
            #J_f = delete_from_csr(J_f.tocsr(), row_indices=jac_row_ind, col_indices=[]) 
            J_l = delete_from_csr(J_c.tocsr(), col_indices=col_ind)  
            
        else:
            None
        
        r_hess, Z_mat = self._reduced_hessian_matrix(J_f, J_l, H, col_ind)
       
        if not return_Z:
            return r_hess.todense()
        else:
            return r_hess.todense(), Z_mat

    @staticmethod
    def _reduced_hessian_matrix(F, L, H, col_ind):
        """This calculates the reduced hessian by calculating the null-space based
        on the constraints
        
        Args:
            F (csr_matrix): Rows of the Jacobian related to fixed parameters
            
            L (csr_matrix): The remainder of the Jacobian without parameters
            
            H (csr_matrix): The sparse Hessian
            
            col_ind (list): indicies of columns with fixed parameters
        
        Returns:
            reduced_hessian (csr_matrix): sparse version of the reduced Hessian
            
        """
        n = H.shape[0]
        n_free = n - F.shape[0]
        
        X = spsolve(L.tocsc(), -F.tocsc())
        
        col_ind_left = list(set(range(n)).difference(set(col_ind)))
        col_ind_left.sort()
        
        Z = np.zeros([n, n_free])
        Z[col_ind, :] = np.eye(n_free)
        
        if isinstance(X, csc_matrix):
            Z[col_ind_left, :] = X.todense()
        else:
            Z[col_ind_left, :] = X.reshape(-1, 1)
            
        Z_mat = coo_matrix(np.mat(Z)).tocsr()
        Z_mat_T = coo_matrix(np.mat(Z).T).tocsr()
        Hess = H.tocsr()
        reduced_hessian = Z_mat_T * Hess * Z_mat
        
        return reduced_hessian, Z_mat

    def add_global_constraints(self):
        """This adds the dummy constraints to the model forcing the local
        parameters to equal the current global parameter values
         
        Args:
            model_object (pyomo ConcreteModel): Pyomo model to add constraints to
            
            parameter_set (list): List of parameters to fix using constraints
            
            scaled (bool): True if scaled, False if not scaled
             
        Returns:
            None
         
        """
        if self.parameter_set is None:
            self.parameter_set = [p for p in getattr(self.model_object, self.variable_name)]
         
        if self.scaled:
            global_param_init = {p: 1 for p in self.parameter_set}
        else:
            global_param_init = {p: getattr(self.model_object, self.variable_name)[p].value for p in self.parameter_set}
    
        for comp in [self.global_constraint_name, self.global_param_name, self.current_set]:
            if hasattr(self.model_object, comp):
                self.model_object.del_component(comp)   
         
        setattr(self.model_object, self.current_set, Set(initialize=self.parameter_set))
    
        setattr(self.model_object, self.global_param_name, 
                Param(getattr(self.model_object, self.param_set_name),
                              initialize=global_param_init,
                              mutable=True,
                              ))
         
        def rule_fix_global_parameters(m, k):
            return getattr(m, self.variable_name)[k] - getattr(m, self.global_param_name)[k] == 0
             
        setattr(self.model_object, self.global_constraint_name, 
                Constraint(getattr(self.model_object, self.current_set),
                           rule=rule_fix_global_parameters)
            )
        
        return None
 
def read_duals(sol_file):
    """Reads the duals from the sol file after solving the problem
    
    Args:
        sol_file (str): The absolute path to the sol file
        
    Returns:
        duals (list): The list of duals values taken from the sol file
    
    """
    sol_file_abs = Path(sol_file)
    
    duals = []
    with sol_file_abs.open() as f: 
        lines = f.readlines()
       
    lines_found = True
    num_of_vars = int(lines[9])
      
    for ln, line in enumerate(lines):
        line = line.rstrip('\n')
        line = line.lstrip('\t').lstrip(' ')
        
        if ln >= 12 and ln <= (num_of_vars + 11):
            duals.append(float(line))
            
    return duals

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by 
    ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    
    Args:
        mat (csr_matrix): Sparse matrix to delete rows and cols from
        
        row_indicies (list): rows to delete
        
        col_indicies (list): cols to delete
        
    Returns:
        mat (csr_matrix): The sparse matrix with the rows and cols removed
    
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

class SparseRowIndexer:
    """Class used to splice sparse matrices"""
    
    def __init__(self, matrix):
        data = []
        indices = []
        indptr = []
        
        _class = 'csr'
        #print(f'LOOK HERE: {type(matrix)}')
        if isinstance(matrix, csc_matrix):
            _class = 'csc'
       #     print(_class)
        
        self._class = _class
        # Iterating over the rows this way is significantly more efficient
        # than matrix[row_index,:] and matrix.getrow(row_index)
        for row_start, row_end in zip(matrix.indptr[:-1], matrix.indptr[1:]):
             data.append(matrix.data[row_start:row_end])
             indices.append(matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]
        
        if self._class == 'csr':
            return csr_matrix((data, indices, indptr), shape=shape)
        else:
            return csr_matrix((data, indices, indptr), shape=shape).T.tocsc()