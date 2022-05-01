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
                            'pEV':rainfall.pEV, 'leachate': leachate.leach}, 
                      index = rainfall.index)
table_final = table.drop(table.index[-1])
print(table_final)

rain = np.asarray(table_final.rain)
evaporation = np.asarray(table_final.pEV)
Qdr_observed = np.asarray(table_final.leachate)

# Definition of parameters
acl = 0.007     # 0.005~0.01
awb = 0.0008    # 0.0005~0.001
Scl_max = 0.60  # 0.45~0.975
Swb_max = 7.0   # 4.5~7.8
Scl_min = 0     
Swb_min = 0
beta0 = 0.98    # 0~1
fcrop = 0.92       # 0~1.4
fred = 1.0
bcl = 5         # 0~80
bwb = 30        # 0~80
S_Evmin = 0
S_Evmax = 1


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
Y0 = np.array([0.8, 0.8])
Q_rate = np.zeros(len(evaporation))
Q_rate[0] =  0.
for i in range (1, nOut-1):
    Q_rate[i] = Qdr_observed[i] - Qdr_observed[i-1]
    
import scipy.integrate as spint
YODE = spint.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )


tOut = np.linspace(0, len(evaporation)-1, len(evaporation))

Qdr_cal = beta0 *((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min))*acl*((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min)) ** bcl + awb*((YODE.y[1,:] - Swb_min)/(Swb_max - Swb_min)) ** bwb
Qdr_cal = Qdr_cal * 28355     #base_area = 28355
SSE = np.sum((Qdr_cal - Qdr_observed)**2)
#YODE = spint.solve_ivp(dydt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
   # sort this Qdr_cal = B * Lcl + Lwb
#SSE = np.sum((Qdr_cal - Qdr_observed)**2)



""" Rate of change of storage in the drainage layer Sdr. """
#def dSdrdt(t, Y): 
    #B(t) * Lcl(t) + Lwd(t) - Qdr(t) = 0


plt.figure(figsize=(9,5))
plt.plot(tOut, Qdr_observed, 'r', markersize=2.5, label='Observations')
plt.plot(tOut, Qdr_cal, 'o', markersize=2.5, label='Calculated')
plt.ylabel('Storage (m)')
plt.title('SSE')
plt.legend()
plt.grid();
