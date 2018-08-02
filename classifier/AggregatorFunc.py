#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:58:09 2018

@author: bingyangwen

Purpose: Aggregator Functions
"""
import numpy as np

def snd_to_eng(w,torque,engine):
    '''
    Algorithm 2 - line 4
    Send torque and w to engines from aggregator
    Var: 
        w: Updated parameter w
        torque: iteration for local updates
    '''
    for i in range(len(engine.ids)):
        engine[i]['w_aggregator'] = w
        engine[i]['torque_aggregator'] = torque
        

def wc_from_eng(engine, object_name):
    '''
    Algorithm 2 - line 7
    Retrieve belta and grad from engines  to aggregator.  
    Var:
        object_name: name of classifier object variable, used for retrieve object information.
        mode parameters saved in local edges. It might be any shape.
    Output:
        w as a vector shape
        w shape = (# of engines, # of parameters)
        c shape = (# of engines,)
    '''
    if not isinstance(object_name,str):
        raise ValueError('Value of object_name must be string, {} is given!'.format(object_name))
            
    w = []
    c = []
    for i in range(len(engine.ids)):
        w.append(engine[i][object_name + '.get_coef()'])
        c.append(engine[i][object_name + '.Est_Resource()']) 
    w = np.array(w)
    c = np.array(c)
        
    if len(w[0].shape) != 1:
        # reshape w to vector
        w = w.reshape(w.shape[0],1,-1)
        
    return w,c


data_size = np.array([50,50,50])
def global_update(w_local,data_size):
    '''
    Algorithm 2 - line 8, global parameter updata according to (5)
    Aggregator parameter updating rules: mean    
    Input: Matrix with shape = (# of engines, # of parameters)
    Output: Matrix with shape = (# of engines, # of parameters)
    '''
    temp = 0
    for i in range(len(w_local)):
        temp = temp + w_local[i] * data_size[i]
    return np.array(temp/data_size.sum())  



def bg_from_eng(engine, object_name):
    '''
    Algorithm 2 - line 14
    Retrieve belta and grad from aggregator 
    Var:
        engine: belta and gradient of classifer object from Engines.
        object_name: name of classifier object variable, used for retrieve object information.
    Output:
        Gradient and belta as vector.
        Gradient shape = (# of engines, # of parameters)
        belta shape = (# of engines,)
    '''
    if not isinstance(object_name,str):
        raise ValueError('Value of object_name must be string, {} is given!'.format(object_name))
        
    belta = []
    grad = []
    for i in range(len(engine.ids)):
        belta.append(engine[i][object_name + '.belta'])
        grad.append(engine[i][object_name + '.grad_t0']) 
    grad = np.array(grad)
    belta = np.array(belta)
    
    if len(grad[0].shape) != 1:
        # reshape grad to vector
        grad = grad.reshape(grad.shape[0],1,-1)
        
    return belta,grad



def belta_update(belta_local,data_size):
    '''
    Algorithm 2 - line 15
    Estimate global belta, updating rules: mean
    Var:
        belta_local: belta recieved from engines. Shape = (# of engines,)
        data_size: data_size(# of samples) that sent to local
        
    Output: Estimated belta (scalar)
    '''
    temp = 0
    for i in range(len(belta_local)):
        temp = temp + np.array(belta_local[i]) * data_size[i]
    return temp/data_size.sum()




def grad_update(grad_local,data_size):
    '''
    Algorithm 2 - line 16
    Var:
        grad_local: gradients collected from local edges
        data_size: data_size(# of samples) that sent to local 
    Output:
        Updated global gradients, shape = (# of parameters,)
        Output as input of delta_update
        
    ''' 
    temp = 0
    for i in range(len(grad_local)):
        temp = temp + np.array(grad_local[i]) * data_size[i]
    return temp/data_size.sum()

def delta_update(grad_local, grad_aggregator, data_size):
    '''
    Algorithm 2 - line 16
    Var:
        grad_aggregator: Output of grad_update.
        grad_local: gradients collected from local edges
        data_size: data_size(# of samples) that sent to local 
    Output:
        Updated global gradients, shape = (1, # of parameters)
    ''' 
    delta_local = []
    grad_local = np.array(grad_local) 
    grad_aggregator = np.array(grad_aggregator)
    for item in grad_local:
        temp_delta = np.linalg.norm(item - grad_aggregator)
        delta_local.append(temp_delta)
    #belta_update() used as a weighted average function
    return belta_update(delta_local,data_size)



def binary_search(torque, delta, belta, gamma,phi,eta = 0.0001):
    '''
    Algorithm 2 - line 17, binary search for new torque.
    Var:
        torque: Old torque, used to calculate new torque
        delta: Output of delta_update()
        belta: Output of belta_update()
        gamma: Parameter to control searching bound
        phi: As a control parameter that is manually chosen and remains fixed for the same machine learning model
        eta: Learning rate
    '''
    upper_bound = int(gamma * torque)
    G_list = []
    for i in range(upper_bound):
        torque_try = i + 1
        G_list.append(G(torque_try, delta, belta, eta, phi))
    torque_star = np.argmax(np.array(G_list)) + 1
    return torque_star, G_list

def G(torque, delta, belta, eta, phi):
    '''
    Algorithm 2 - line 17, binary search for new torque.
    '''
    h = delta* (pow((eta*belta + 1),torque)-1)/belta - eta*delta*torque
    G = torque*(eta*(1-belta*eta/2)-phi*h/torque)
    return G

