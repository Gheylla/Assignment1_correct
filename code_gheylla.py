# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:37:08 2022

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
#import MyTicToc as mt

import pandas as pd

wieringermeer_leachate = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\Assignment1_correct\WieringermeerData_LeachateProduction.xlsx')
wieringermeer_meteo = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\Assignment1_correct\WieringermeerData_Meteo (1).xlsx')


plt.plot(wieringermeer_leachate.Leachate)
plt.ylabel("Leachate production in m3/day")
plt.title("Leachate Production")
plt.grid()

beta = 1
J = 1

L_cl =[10,10]
L_wd = [10, 10]
E = 10

#differential functions 

def dYdt(t, Y):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ J - L_cl[0] - E ,
                     (1 - beta) * L_cl[0] - L_wd[1]])









