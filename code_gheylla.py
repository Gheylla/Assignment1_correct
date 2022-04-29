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
wieringermeer_leachate = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\WieringermeerData_LeachateProduction.xlsx')
wieringermeer_meteo = pd.read_excel(r'C:\Users\User\Desktop\TU DELFT\Master 2021-2022\Q4\Modelling for Coupled Processes\WieringermeerData_Meteo.xlsx')

#exclude = wieringermeer_meteo[wieringermeer_meteo['Date'].dt.year != year]


plt.plot(wieringermeer_leachate[0])
plt.ylabel("Leachate production in m3/day")
plt.title("Leachate Production")
plt.grid()

print(wieringermeer_meteo.columns)


#define all the variables from the excel files
#Q_dr = wieringermeer_meteo[1]
Jrf = wieringermeer_meteo.loc[:,'rain_station']
pEV = wieringermeer_meteo.loc[:, 'pEV']
temp = wieringermeer_meteo.loc[:, 'temp']


#define the constants 

beta = 1
L_cl =[10,10]
L_wd = [10, 10]
E = 10

#differential functions 

def dYdt(t, Y):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([ J - L_cl[0] - E ,
                     (1 - beta) * L_cl[0] - L_wd[1]])


def main():
    # Definition of output times
    tOut = np.linspace(0, 100, 200)              # time
    nOut = np.shape(tOut)[0]

    # Initial case, 10 rabbits, 5 foxes
    Y0 = np.array([10, 5])
    mt.tic()
    t_span = [tOut[0], tOut[-1]]
    YODE = sp.integrate.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, 
                                  method='RK45', vectorized=True, 
                                  rtol=1e-5 )
    # infodict['message']                     # >>> 'Integration successful.'
    rODE = YODE.y[0,:]
    fODE = YODE.y[1,:]

    '''EULER'''
    # Initialize output vector
    YEuler = np.zeros([nOut, 2], dtype=float)

    dtMax = 0.1
    # dtMin = 1e-11
    t = tOut[0]
    iiOut = 0

    # Initialize problem
    mt.tic()
    Y = Y0
    # Write initial values to output vector
    YEuler[iiOut, [0, 1]] = Y
    while (t < tOut[nOut-1]):
        # check time steps
        Rates = dYdt(t, Y)
        dtRate = -0.7 * Y/(Rates + 1e-18)
        dtOut = tOut[iiOut+1]-t
        dtRate = (dtRate <= 0)*dtMax + (dtRate > 0)*dtRate
        dt = min(min(dtRate), dtOut, dtMax)

        Y = Y + Rates * dt
        t = t + dt

        # print ("Time to Output is " + str(np.abs(tOut[iiOut+1]-t)) +
        # " days.")

        if (np.abs(tOut[iiOut+1]-t) < 1e-5):
            YEuler[iiOut+1, [0, 1]] = Y
            iiOut += 1

    rEuler, fEuler = YEuler.T
    mt.toc()




