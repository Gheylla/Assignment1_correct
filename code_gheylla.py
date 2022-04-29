# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:37:08 2022

@author: Gheylla
"""

import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
# MyTicToc as mt

import pandas as pd

#import excel data
#remove the years from 2003 -2011 in order to make the number
#of values the same for both excel sheets. 

wieringermeer_leachate = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\WieringermeerData_LeachateProduction.xlsx')
wieringermeer_meteo = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\WieringermeerData_Meteo.xlsx', skiprows = range(1,3453))
wieringermeer_meteo = wieringermeer_meteo.drop(wieringermeer_meteo.index[-1])


plt.plot(wieringermeer_leachate[0])
plt.ylabel("Leachate production in m3/day")
plt.title("Leachate Production")
plt.grid()


#define all the variables from the excel files
Q_dr = wieringermeer_leachate.iloc[:, 1]
Jrf = wieringermeer_meteo.loc[:,'rain_station']
pEV = wieringermeer_meteo.loc[:, 'pEV']
temp = wieringermeer_meteo.loc[:, 'temp']

#define the constants 
# Definition of parameters
S_Evmax = 1
S_Evmin = 0.25
Scl_min = 0.1
Scl_max = 1
Swb_max = 1 
Swb_min = 0.1
a = 0.1                         #saturated hydraulic conductivity [m/day]
b_cl = 0.01                      # empirical parameter
b_wb = 0.01
base_area = 28355               #[m2]
top_area = 9100                 #[m2]
slope_width = 38                #[m]
waste_body_height = 12          #[m]
cover_layer_height = 1.5        #[m]
waste = 281083000               #wet weight [kg]
f_crop = 1                          #Crop factor
Bo = 1
L_wd = 2
L_cl = 1
beta = 2
f_red = 1

#differential functions 
def dYdt(t, Y):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ Jrf - L_cl - E ,
                     (1 - beta) * L_cl - L_wd])
                    
#define the other functions 
#def dSdrdt(t, Y): 
 #   return beta * L_cl + L_wd - Q_dr = 0 
 
E = pEV * f_crop * f_red

def main():
    # Definition of output times
    tOut = np.linspace(0, 100, 200)              # time
    nOut = np.shape(tOut)[0]

    # Initial case, 10 rabbits, 5 foxes
    Y0 = np.array([10, 5])
    #mt.tic()
    t_span = [tOut[0], tOut[-1]]
    YODE = sp.integrate.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, 
                                  method='RK45', vectorized=True, 
                                  rtol=1e-5 )
    # infodict['message']                     # >>> 'Integration successful.'
    rODE = YODE.y[0,:]
    fODE = YODE.y[1,:]







