# -*- coding: utf-8 -*-
"""
Created on Fri Aug  28 10:15:12 2020

@author: Benjamin.Garzon
"""

# change color of attributes

# how are the contexts selected
#  how many choices should I generate to cover everything?

# probability of getting selected is proportional to difference in attribute
# generate values ?, different functions of the features
# the sequence of choices should be wrt attribute
# plot shirts on distribution
# add a different item
# one item does not have both attributes
# get ratings
# manipulate delay of attributes and number of repeats

# generate correct combinations in a smart way xxx
# training and testing phases xxxx
# document

import numpy as np
from funcs import *

a = 1
b = -0.5
c = 1/8 # equal weights for both attributes
d = 10

nitems = 8
nreps = 3
nchoices = 0
nrounds = 2

###############################################################################
# Define value functions and parameters
###############################################################################

weights1 = np.array([np.cos(2*np.pi*c), np.sin(2*np.pi*c),  10])
value_function1 = generate_value_function(weights1)

value_function2 = value_function1


params1 = {'context_name' : 'tshirts',
           'context_image' : './images/tshirt2.jpg',
           'attr_names' : ('ABC Stores', 'CBA Stores'),
           'attr_colours' : ('red', 'blue'),
           'mean' : np.array([5, 5]),
           'cov' : np.array([[a**2, a*b], [a*b, 1]]),
           'value_function' : value_function1 }

params2 = {'context_name' : 'shorts',
           'context_image' : './images/shorts2.jpg',
           'attr_names' : ('ABC Stores', 'CBA Stores'),
           'attr_colours' : ('red', 'blue'),
           'mean' : np.array([5, 5]),
           'cov' : np.array([[a**2, a*b], [a*b, 1]]),
           'value_function' : value_function2 }

###############################################################################
# Define contexts
###############################################################################

context1 = Context(params1, nitems)
context2 = Context(params2, nitems)

choice_table_context1_attr1 = context1.generate_binary_choices(nchoices, 
                                                               which_attr = 0, 
                                                               nreps = nreps, 
                                                               nrounds = nrounds)
choice_table_context1_attr2 = context1.generate_binary_choices(nchoices, 
                                                               which_attr = 1, 
                                                               nreps = nreps,
                                                               nrounds = nrounds)
choice_table_context2_attr1 = context2.generate_binary_choices(nchoices, 
                                                               which_attr = 0, 
                                                               nreps = nreps, 
                                                               nrounds = nrounds)
choice_table_context2_attr2 = context2.generate_binary_choices(nchoices, 
                                                               which_attr = 1, 
                                                               nreps = nreps,
                                                               nrounds = nrounds)

choice_table_context1_attr1['Session_Type'] = 'training'
choice_table_context1_attr2['Session_Type'] = 'training'
choice_table_context2_attr1['Session_Type'] = 'training'
choice_table_context2_attr2['Session_Type'] = 'training'

choice_table_context1_attr1.to_csv('./choices/choice_table_context1_attr1.csv', 
                                   float_format='%.3f')
choice_table_context1_attr2.to_csv('./choices/choice_table_context1_attr2.csv', 
                                   float_format='%.3f')
choice_table_context1_attr2.to_csv('./choices/choice_table_context2_attr1.csv', 
                                   float_format='%.3f')
choice_table_context1_attr2.to_csv('./choices/choice_table_context2_attr2.csv', 
                                   float_format='%.3f')

    
