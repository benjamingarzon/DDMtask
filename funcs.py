# -*- coding: utf-8 -*-
"""
Created on Fri Aug  28 10:15:12 2020

@author: Benjamin.Garzon
"""
from fractal import Fractal

import random, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from itertools import combinations, cycle
import cv2
from PIL import Image

# Image resolution
DPI = 500

def myrng(mu, sigma, n):
    return(np.random.multivariate_normal(mu, sigma, n))

def generate_value_function(weights):
    """
    Returns a value function based on weights passed.
    """    
    f = lambda x: weights.dot(x) 
    return(f)

class Item():
    """
    Defines items to be chosen.
    """
    def __init__(self, attr, context_name, value_function, attr_names, 
                 background_image, index):
        self.index = index
        self.attr = attr
        self.attr_names = attr_names
        self.name = '%s-%0.3d'%(context_name, index)
        self.image_file = './images/items/%s.png'%(self.name)
        self.value = value_function(np.array(list(attr) + [1.0]))
        self.marginal_value = (
            value_function(np.array([attr[0], 0.0, 1.0])),
            value_function(np.array([0.0, attr[1], 1.0]))
            )
        
        background = cv2.imread(background_image)
        background = 255*(background < 100).astype('uint8')

        if not os.path.exists(self.image_file):
            
            f = Fractal()
            f.generate(self.image_file, 
                       width = background.shape[0],
                       height = background.shape[1])
            myimage = cv2.imread(self.image_file)
            myimage = cv2.bitwise_and(background, myimage)
            myimage[background == 0] = 130
            cv2.imwrite(self.image_file, myimage)


class Distribution():
    """
    Consider 2D Gaussian distribution only.
    """
    def __init__(self, params, context_name, value_function):
        self.attr_names = params['attr_names'] # attribute name
        self.params = params # distribution parameters
        self.context_name = params['context_name']
        self.context_image = params['context_image']
        self.value_function = value_function
        
    def plot(self):
        """
        Plot distribution and samples.
        """
       
        mu = self.params['mean']
        sigma = self.params['cov']
        sd = np.sqrt(np.diag(sigma))

        X, Y = np.meshgrid(np.linspace(mu[0] - 4*sd[0], 
                                       mu[0] + 4*sd[0], 100), 
                           np.linspace(mu[1] - 4*sd[1], 
                                       mu[1] + 4*sd[1], 100))
        pos = np.dstack((X, Y))

        f = multivariate_normal(mu,
                                sigma).pdf
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(X, Y, f(pos), levels = 100, cmap = "bone")
        plt.axis('equal')
        self.fig = fig
        
    def save_plot(self, item_list = None):
        
        if item_list:
            x, y, z = zip(*[(item.attr[0], 
                           item.attr[1], 
                           item.value) for item in item_list])
            
            plt.figure(self.fig.number)
            plt.scatter(x, 
                    y, 
                    c = z, 
                    s = 30, 
                    cmap = "autumn",  
                    marker=".", 
                    vmin = np.min(z), 
                    vmax = np.max(z))
            plt.colorbar()

        image_file = './images/distributions/%s.png'%(self.context_name)
        self.fig.savefig(image_file, dpi = DPI)
        
    def generate_items(self, n, plot = True):
        """
        Generate n samples from the distribution
        """
        x, y = myrng(self.params['mean'], 
                     self.params['cov'], 
                     n).T
        
        items = []
        for i in range(n):
            items.append(Item((x[i], y[i]), 
                              self.context_name, 
                              self.value_function,
                              self.params['attr_names'],
                              self.context_image,
                              i))        
        # Make a pic with all the possible items
        if plot:
            size = 200
            k = int(np.floor(len(items) ** 0.5))
            sizey = size*k #int(size*np.ceil(np.sqrt(len(items))))
            sizex = size*int(np.ceil(len(items)/k)) 
            new_im = Image.new('RGB', (sizex, sizey))
            index = 0

            
            for i in range(0, sizex, size):
                for j in range(0, sizey, size):
                    im = Image.open(items[index].image_file)
                    im.thumbnail((size, size))
                    new_im.paste(im, (i, j))
                    index += 1
                        
            new_im.save('./images/items/all-%s.png'%(self.context_name))

        return(items)

class Context():

    def __init__(self, params, n):
                
        self.context_name = params['context_name']
        self.context_image = params['context_image']
        self.attr_names = params['attr_names'] # attribute name
        self.attr_colours = params['attr_colours'] 
        self.value_function = params['value_function']
        self.distribution = Distribution(params, 
                                         self.context_name, 
                                         self.value_function)  
        self.distribution.plot()
        self.item_list = self.distribution.generate_items(n)
        
        #self.item_short_list = sample nn
        #self.presentation_text = 
        #self.presentation_image = 
        self.distribution.save_plot(self.item_list)
        
           
    def generate_binary_choices(self, k, which_attr, nreps = 1, nrounds = 1):
        """
        Creates a list with binary choices of items sampled from item_list.
        """
        n = len(self.item_list)
        comb_list = list(combinations(range(n), 2))
        ncombs = len(comb_list)
        
        print("There are %d possible "%(ncombs) + \
              "combinations of 2 elements from a " + \
              "total of %d elements."%(n))

        # shuffle across subjects
        self.choices = []
        choice_table = []
        i = 0
        if k == 0:
            k = ncombs
        
        for myround in range(nrounds):
            random.shuffle(comb_list)
            combs = cycle(comb_list)

            while i < k:
                # draw from list
                # check not there
                mycomb = next(combs)
                item1 = self.item_list[mycomb[0]]
                item2 = self.item_list[mycomb[1]]  
                choice = (item1, item2)
                self.choices.append(choice)
                
                correct_item = 1 \
                    if item1.attr[which_attr] > item2.attr[which_attr] else 2
    
                for r in range(nreps):
                    switch_side = np.random.choice([0, 1]) #if 0 item1 on the left
    
                    if switch_side == 0:
                        image_left = os.path.abspath(item1.image_file)
                        image_right = os.path.abspath(item2.image_file)
                        correct_answer = 'left' \
                            if correct_item == 1 else 'right'
                    else:                    
                        image_left = os.path.abspath(item2.image_file)
                        image_right = os.path.abspath(item1.image_file)
                        correct_answer = 'right' \
                            if correct_item == 1 else 'left'
                    choice_table.append([
                                     self.context_name,
                                     i, 
                                     r,
                                     myround,
                                     self.attr_names[which_attr],
                                     self.attr_colours[which_attr],
                                     item1.name,
                                     item1.attr[which_attr],
                                     item1.value,
                                     item1.marginal_value[which_attr],
                                     item2.name,
                                     item2.attr[which_attr],
                                     item2.value,
                                     item2.marginal_value[which_attr],
                                     switch_side,
                                     correct_item,
                                     correct_answer, 
                                     image_left,
                                     image_right
                                     ])
                i = i + 1
            
        choice_table = pd.DataFrame(choice_table, 
                                    columns = ['Context_Name',
                                               'Choice_Index',
                                               'Choice_Rep',
                                               'Round',
                                               'Attr_Name',
                                               'Attr_Colour',
                                               'Item1_Attr',
                                               'Item1_Name',
                                               'Value_Item1', 
                                               'Marg_Value_Item1_Attr', 
                                               'Item2_Name',
                                               'Item2_Attr',
                                               'Value_Item2', 
                                               'Marg_Value_Item2_Attr', 
                                               'Switch_Side',
                                               'Correct_Item',
                                               'Correct_Answer',
                                               'Filename_Left', 
                                               'Filename_Right'
                                               ]
                                    )
        
        return(choice_table)
    
class Schedule():
    """
    Generate different and sessions and contexts within them.
    Randomise for different subjects, and contexts.
    Parameters:
        a = std of x, (0, 3)
        b = correlation x, y (-1, 1)
        c = angle for relationship between weights (0, 1)
        d = delay
        
        # context_name, parameters, session, subject, session_type
    """
#    def __init__(self, attr, context_name, value_function, attr_names, 

                 
