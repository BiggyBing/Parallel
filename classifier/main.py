#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:23:25 2018

@author: bingyangwen
"""
import numpy as np
import ipyparallel as ipp
import time
import tensorflow as tf
from AggregatorFunc import snd_to_eng, wc_from_eng, global_update, bg_from_eng, belta_update, grad_update, delta_update, binary_search
from linearRegression import LinearRegression

if __name__ == '__main__' :
    c = ipp.Client()
    #create a directview
    dview = c[:]
    c[0]['X'] = train[:50]
    c[1]['X'] = train[50:100]
    c[2]['X'] = train[100:150]

    c[0]['y'] = target[:50]
    c[1]['y'] = target[50:100]
    c[2]['y'] = target[100:150]
    
    lr = LinearRegression()
    dview.push(dict(LinearRegression = LinearRegression))
    #sending LinearRegression object to engines
    dview['lr'] = lr
    t = 0
    s = 0
    b = 0 # aggregator consumption
    R = 5
    gamma = 10
    phi = 0.2
    w_aggregator = np.zeros(4)
    torque_aggregator = 1
    stop = False
    
    while True:
        tic = time.time()
        snd_to_eng(w_aggregator,torque_aggregator,c)
        t_0 = t
        t = t + torque_aggregator
        dview.execute("""
import numpy as np
import time    
lr.Rec_from_Agg(w_aggregator, torque_aggregator)
lr.time_record()
if lr.t > 0:
    lr.Est_Belta(X,y)
lr.fit(X,y)
        """)
        
        
        w_local, c_local = wc_from_eng(c,'lr')
        #calculate local consumption per iteration
        c_per = np.array(c_local)/torque_aggregator
        w_aggregator = global_update(w_local, data_size)
        if stop:
            w_final = w_aggregator
            break
        #c_local.sum() equal to c*t
        s = s + np.array(c_local).sum()+ b
        
        if t_0 > 0:
            belta_local, grad_local = bg_from_eng(c)
            belta_aggregator = belta_update(belta_local,data_size)
            grad_aggregator = grad_update(grad_local,data_size)
            delta_aggregator = delta_update(grad_local, grad_aggregator, data_size)
            torque_aggregator, G_list = binary_search(torque_aggregator, delta_aggregator, belta_aggregator, gamma, phi)
            print('New torque is:',torque_aggregator)
        
        toc = time.time()
        b = toc - tic
        temp = s + torque_aggregator*c_per.sum() + b
        if temp >= R:
            torque_max = (R-b-s)/c_per.sum()
            G_list = np.array(G_list)
            G_min = G_list.min()
            for i, item in enumerate(G_list):
                if item >= torque_max:
                    itme = G_min
            torque_aggregator = np.argmax(G_list) + 1
            stop = True 