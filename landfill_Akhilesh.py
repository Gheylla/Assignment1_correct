import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

#Import the required files
leachate = pd.read_excel('WieringermeerData_LeachateProduction.xlsx', index_col=0)
meteo = pd.read_excel('WieringermeerData_Meteo.xlsx', index_col=0)
rainfall = meteo.loc[meteo.index >= "2012-06-14"]
leachate.columns = ['Cumulative_leach', 'leach']
table = pd.DataFrame(data={'rain':rainfall.rain_station,
                            'pEV':rainfall.pEV, 'leachate': leachate.leach})
table_final = table.drop(table.index[-1])

base_area = 28355
table_final['leachate'] = table_final['leachate']/base_area

print(table_final)


rain = np.asarray(table_final.rain)
evaporation = np.asarray(table_final.pEV)
Qdr_observed = np.asarray(table_final.leachate)

# Definition of parameters
acl = 0.005     # 0.005~0.01
awb = 0.001   # 0.0005~0.001
Scl_max = 0.60  # 0.45~0.975
Swb_max = 6.0   # 4.5~7.8
Scl_min = 0.02     
Swb_min = 0.02
beta0 = 0.98    # 0~1
fcrop = 0.85       # 0~1.4
fred = 1.0
bcl = 4         # 0~80
bwb = 30        # 0~80
S_Evmin = 0
S_Evmax = 0.01


def dYdt(t, Y):
    Lcl = ((Y[0] - Scl_min)/(Scl_max - Scl_min))
    Lwb = (Y[1] - Swb_min)/(Swb_max - Swb_min)
        
    #Boundaries for values of storage
    if Y[0] < Scl_min:                  
        Y[0] = Scl_min
    elif Y[0] > Scl_max:
        Y[0] = Scl_max   
    if Y[1] < Swb_min:
            Y[1] = Swb_min
    elif Y[1] > Swb_max:
        Y[1] = Swb_max 
            
    #reduction factor reducing evapotranspiration under dry soil conditions

    if Y[0] < S_Evmin:
        fred = 0
    elif S_Evmin <= Y[0] <= S_Evmax:
        fred = (Y[0] - S_Evmin) / (S_Evmax - S_Evmin)
    else: 
        fred = 1

    dydt = np.zeros(2)
    dydt[0] = (rain[int(t)]) - acl*(Lcl)**bcl - (evaporation[int(t)])*fcrop*fred
    dydt[1] = (1 - beta0*Lcl) * acl*Lcl**bcl - awb*(Lwb)**bwb 
    return dydt
    
tOut = np.linspace(0, len(evaporation)-1, len(evaporation))              
nOut = np.shape(tOut)[0]
t_span = [tOut[0], tOut[-1]]

Y0 = np.array([0.5, 0.5])

import scipy.integrate as spint
YODE = spint.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )

Qdr_cal = np.zeros(len(evaporation))
Qdr_cal = beta0 *((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min))*acl*((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min)) ** bcl + awb*((YODE.y[1,:] - Swb_min)/(Swb_max - Swb_min)) ** bwb


#comparison
Qdr_obs_cum = np.zeros(len(rain))
Qdr_cal_cum = np.zeros(len(rain))
Qdr_cal_cum[0] = Qdr_cal[0]
Qdr_obs_cum[0] = Qdr_observed[0]
for i in range (1, nOut-1):
    Qdr_obs_cum[i] = Qdr_obs_cum[i-1] + Qdr_observed[i]
    Qdr_cal_cum[i] = Qdr_cal_cum[i-1] + Qdr_cal[i]

#Senstivity
S = np.sum((Qdr_cal - Qdr_observed)**2)

#Drainage available
plt.figure(figsize=(9,5))
plt.plot(tOut, Qdr_observed, 'r-', markersize=2.5, label='Observations')
plt.plot(tOut, Qdr_cal, 'b-', markersize=2.5, label='Calculated')
plt.xlabel('Time (days)')
plt.ylabel('Storage (m)')
plt.title('Storage available')
plt.legend()
plt.grid();

#Storage cumulative
plt.figure(figsize=(9,5))
plt.plot(tOut, Qdr_cal_cum, label='Modelled cumulative curve')
plt.plot(tOut, Qdr_obs_cum, label='Observed cumulative curve')
plt.ylabel('Cumulative Drainage')
plt.legend()
plt.grid();

# Plot Cover layer storage and Waste body storage over time
plt.figure(figsize=(9,5))
plt.ylabel('Storage [m]')
plt.plot(tOut, YODE.y[0,:], label='Cover layer', color ='blue')
plt.plot(tOut, YODE.y[1,:], label='Waste layer', color ='green')
plt.legend()
plt.grid();

#water balance
plt.figure(figsize=(9,5))
plt.plot(tOut, rain, label='Rainfall')
plt.plot(tOut, Qdr_cal, label='Drainage')
plt.plot(tOut, evaporation, label='pEV')
plt.ylabel('[m/day]')
plt.title('Total Water Balance')
plt.legend()
plt.grid();

#senstivity?
plt.figure(figsize=(9,5))
plt.plot(tOut, S, label='Senstivity (Calculated and Observed)', color ='blue')
plt.ylabel('Drainage[m]')
plt.legend()
plt.grid();
