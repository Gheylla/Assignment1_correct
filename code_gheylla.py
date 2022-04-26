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

wieringermeer_meteo = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\Assignment1_correct\WieringermeerData_LeachateProduction.xlsx')
wieringermeer_leachate = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\Assignment1_correct\WieringermeerData_Meteo (1).xlsx')

#display(wieringermeer_meteo)



plt.plot(wieringermeer_leachate)
plt.xlabel("days")
plt.ylabel("Leachate production in m3/day")
plt.title("Leachate Production")
plt.grid()








