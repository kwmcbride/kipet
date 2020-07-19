#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:16:01 2020

@author: kevin
"""

from pyomo.core.expr import current as EXPR
from pyomo.environ import *

class ReplacementVisitor(EXPR.ExpressionReplacementVisitor):
    def __init__(self):
        super(ReplacementVisitor, self).__init__()
        self._replacement = None
        self._suspect = None

    def change_suspect(self, suspect_):
        self._suspect = suspect_
        
    def change_replacement(self, replacement_):
        self._replacement = replacement_

    def visiting_potential_leaf(self, node):
        #
        # Clone leaf nodes in the expression tree
        #
        if node.__class__ in native_numeric_types:
            return True, node

        # if node.__class__ is NumericConstant:
        #     return True, node

        if node.is_variable_type():
            if id(node) == self._suspect:
                d = self._replacement
                return True, d
            else:
                return True, node
            
        return False, None
    
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
    
# This needs to work for the models and not an instance of whatever this was
def scale_parameters(self):
    """If scaling, this multiplies the constants in model.K to each
    parameter in model.P.
    
    I am not sure if this is necessary and will look into its importance.
    """
    for k, model in self.models_dict:
    
        #if self.model.K is not None:
        self.scale = {}
        for i in self.model.P:
            self.scale[id(self.model.P[i])] = self.model.P[i]
    
        for i in self.model.Z:
            self.scale[id(self.model.Z[i])] = 1
            
        for i in self.model.dZdt:
            self.scale[id(self.model.dZdt[i])] = 1
            
        for i in self.model.X:
            self.scale[id(self.model.X[i])] = 1
    
        for i in self.model.dXdt:
            self.scale[id(self.model.dXdt[i])] = 1
    
        for k, v in self.model.odes.items(): 
            scaled_expr = self.scale_expression(v.body, self.scale)
            self.model.odes[k] = scaled_expr == 0
        
def scale_expression(self, expr, scale):
    
    visitor = ScalingVisitor(scale)
    return visitor.dfs_postorder_stack(expr)
    