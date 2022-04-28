# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 13:43:29 2022
Landfill_WaterBalance_Model
@author: mazen
"""
##Step 1: Importing necessary Libraries
import numpy as np
import math
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.integrate as spint

##Step 2: Read Meteo and measured data
#Meto Data
meteo_data = pd. read_excel('WieringermeerData_Meteo.xlsx', index_col=0)

#Measured Data:
meas_data = pd.read_excel('WieringermeerData_LeachateProduction.xlsx', index_col=0)

##Step 3: Match the measured and meto data in term of date and time. 
    #the same date for leachate (Qdr), Rainfall ((Jrf), and Evaporation (E) 
    #using loc and iloc function

Qdr = meas_data.iloc[:, 0]                   # leachate output in [m^3/day]
Jrf = meteo_data.iloc[-(len(Qdr) + 1) : -1, 1]      # Precipitation [m/day]
pE = meteo_data.iloc[-(len(Qdr) +1): -1, 2]       # Evaporation  [m/day]





