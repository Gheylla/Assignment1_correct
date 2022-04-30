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
%matplotlib inline
#Homemade version of matlab tic and toc functions
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

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

##Definition of parameters:
    #Cover layer
acl = 0.007     # 0.005~0.01 [-]
Scl_max = 0.65  # 0.45~0.975 [-]
Scl_min = 0.2
bcl = 8       # 0~80 [m/d] 
beta0 = 0.98    # 0~1
    #Water Balance
awb = 0.0008    # 0.0005~0.001
Swb_max = 0.5  # 4.5~7.8     
Swb_min = 0
Cf = 0.92       # 0~1.4
bwb = 7        # 0~80
fred = 1.0

iniScl = 0.8
iniSwb = 0.8

## Step 4: # Definition of Rate Equations

def dYdt(t, Y):
    """ Rate of change for storage in cover layer Scl and 
    for storage in the waste layer Swb. """
    #a = np.ceil(t)
    
    # Assign each ODE to a vector element
    Scl = Y[0]
    Swb = Y[1]
    
    a = math.ceil(t)
    
    Lcl_rate = acl * ((Y[0] - Scl_min) / (Scl_max - Scl_min)) ** bcl
    
    # Evaporation model
    E_rate = np.array(pE[a-1] * Cf * fred)
    
    # B(t) term that allows a certain fraction of water leaching from the cover layer to directly enter the drainage layer
    beta = beta0 * ((Scl - Scl_min) / (Scl_max - Scl_min))
    Lwb_rate = awb * (((Swb - Swb_min) / (Swb_max - Swb_min))** bwb)       
    
    return np.array([Jrf(a-1) - Lcl_rate - E_rate, 
                     1 - beta * Lcl_rate - Lwb_rate])

#Initial values

Y0 = np.array([0.8, 0.8])

# Definition of output times
tOut = np.linspace(0 , 2757 , 2757)
nOut = np.shape(tOut)[0]

# Solution suing Built-in Solver

t_span = [tOut[0], tOut[-1]]

YODE = spint.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, vectorized=True, 
                       method = 'RK45' , rtol=1e-5)

SclODE = YODE.y[0 , :]
SwbODE = YODE.y[1 , :]

# #Ploting the Figures
# Plot Cover layer storage and Waste body storage over time
plt.figure()
plt.plot(tOut1, SclODE, 'r-', label='Cover layer')
plt.plot(tOut1, SwbODE  , 'b-', label='Waste body')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Time (day)')
plt.ylabel('Water storage (m)')

Qdr_simulated = (beta * Lcl_rate + Lwb_rate)*28355
tOut = np.linspace(0, 2757, 2757)
nOut = np.shape(tOut)[0]
Qdr_simulated = np.zeros(2757)
Qdr_measured = np.zeros(2757)
pE = np.zeros(2757)
Qdr_measured[0] =  0.
# Calculate the storage rate from the measured data
for i in range (1, nOut-1):
    Qdr_measured[i] = Qdr_measured[i] - Qdr_measured[i-1]
print(Qdr_simulated)

plt.figure()
plt.plot(tOut, Qdr_simulated, 'r-', label='Calculated')
plt.plot(tOut, Qdr_measured, 'b-', label='Measured')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Time (day)')
plt.ylabel('Leachate production rate (m^3/day)')
plt.show()
# Plot the calculated (simulated) and measured leachate production rate over time