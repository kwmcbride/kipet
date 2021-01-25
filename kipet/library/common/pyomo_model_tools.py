"""
Model tools
"""

from pyomo.core.base.var import Var
from pyomo.core.base.param import Param
from pyomo.dae import ContinuousSet
from pyomo.dae.diffvar import DerivativeVar


# Pyomo version check
try:
    from pyomo.core.base.set import SetProduct, SimpleSet
except:
    pass
    #print('SetProduct not found')
    
try:
    from pyomo.core.base.sets import _SetProduct, SimpleSet
except:
    pass
    #print('_SetProduct not found')    

def get_vars(model):
    """Extract the variable information from the Pyomo model"""
    
    vars_list = []
    
    model_Var = list(model.component_map(Var))
    model_dVar = list(model.component_map(DerivativeVar))
    
    vars_list = model_Var + model_dVar

    #print(vars_list)

    return vars_list

def get_vars_block(instance):
    """Alternative method for getting the model varialbes"""
    
    model_variables = set()
    for block in instance.block_data_objects():
        block_map = block.component_map(Var)
        for name in block_map.keys():
            model_variables.add(name)
        
    return model_variables

def get_params(instance):
    
    param_list = list(instance.component_map(Param))
    #print(param_list)
    
    return param_list

def get_result_vars(model):
    """Get the Vars and Params needed for the results object"""
    
    result_vars = get_vars(model)
    result_vars += get_params(model)
    return result_vars


def get_index_sets(model_var_obj):
    """Retuns a list of the index sets for the model variable
    
    Args:
        model (ConcreteModel): The pyomo model under investigation
        
        var (str): The name of the variable
        
    Returns:
        
        index_set (list): list of indecies
    
    """
    index_dict = {}
    index_set = []
    index = model_var_obj.index_set()
    
    if not isinstance(index, _SetProduct):
        index_set.append(index)
    
    elif isinstance(index, _SetProduct) or isinstance(index, SetProduct):
        index_set.extend(index.set_tuple)
    
    else:
        return None
    
    return index_set

def index_set_info(index_list):
    """Returns whether index list contains a continuous set and where the
    index is
    
    Args:
        index_list (list): list of indicies produced by get_index_sets
        
    Returns:
        cont_set_info (tuple): (Bool, index of continuous set)
        
    """
    # index_list = a
    
    index_dict = {'cont_set': [],
                  'other_set': [],
                  }
    
    # print(index_list)
    
    for i, index_set in enumerate(index_list):
        if isinstance(index_set, ContinuousSet):
            index_dict['cont_set'].append(i)
        else:
            index_dict['other_set'].append(i)
        
    index_dict['other_set'] = tuple(index_dict['other_set'])
        
    return index_dict