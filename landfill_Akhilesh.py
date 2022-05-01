import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate as spint
import pandas as pd

#Import the required files
leachate = pd.read_excel('WieringermeerData_LeachateProduction.xlsx', index_col=0)
meteo = pd.read_excel('WieringermeerData_Meteo.xlsx', index_col=0)
rainfall = meteo.loc[meteo.index >= "2012-06-14"]
leachate.columns = ['CUM_leach', 'leach']
table = pd.DataFrame(data={'rain':rainfall.rain_station,
                            'pEV':rainfall.pEV, 'leachate': leachate.leach}, 
                      index = rainfall.index)
table_final = table.drop(table.index[-1])
print(table_final)

rain = np.asarray(table_final.rain)
evaporation = np.asarray(table_final.pEV)
Qdr_observed = np.asarray(table_final.leachate)

def variables(x):
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
    x = [acl, awb, Scl_max, Swb_max, Scl_min, Swb_min, beta0, fcrop, fred, bcl, bwb, S_Evmin, S_Evmax]
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
    
    tOut = np.linspace(0, len(evaporation), len(evaporation))              
    nOut = np.shape(tOut)[0]
    Y0 = np.array([Scl_max * 0.4, Swb_max*0.92])
    t_span = [tOut[0], tOut[-1]]
    YODE = sp.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
    Qdr_cal = beta0 *((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min))*acl*((YODE.y[0,:] - Scl_min)/(Scl_max - Scl_min)) ** bcl + awb*((YODE.y[1,:] - Swb_min)/(Swb_max - Swb_min)) ** bwb
    
    SSE = np.sum((Qdr_cal - Qdr_observed)**2)
    return  YODE, Qdr_cal, SSE

base_area = 28355
table_final['leacheate'] = table_final['leachate']/base_area

# Initial case
Y0 = np.array([0.8, 0.8])

# Definition of output times
tOut = np.linspace(0, len(evaporation), len(evaporation))              # time
nOut = np.shape(tOut)[0]
t_span = [tOut[0], tOut[-1]]

Q_rate = np.zeros(len(evaporation))
Q_rate[0] =  0.
for i in range (1, nOut-1):
    Q_rate[i] = Qdr_observed[i] - Qdr_observed[i-1]

#YODE = spint.solve_ivp(dydt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
   # sort this Qdr_cal = B * Lcl + Lwb
#SSE = np.sum((Qdr_cal - Qdr_observed)**2)



""" Rate of change of storage in the drainage layer Sdr. """
#def dSdrdt(t, Y): 
    #B(t) * Lcl(t) + Lwd(t) - Qdr(t) = 0

# Plot results with matplotlib
plt.figure()
plt.plot(tOut, rODE, 'r-', label='Cover layer')
plt.plot(tOut, fODE  , 'b-', label='Waste body')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Time (day)')
plt.ylabel('Water storage (m)')


plt.figure(figsize=(9,5))
plt.plot(table_final.index, Qdr_observed, 'r', markersize=2.5, label='Observations')
plt.plot(table_final.index, bcl, label='Model')
plt.ylabel('Storage (m)')
plt.title(f'SSE: {a}')
plt.legend()
plt.grid();

plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.title('Evolution of fox and rabbit populations')
# f1.savefig('rabbits_and_foxes_1.png')
plt.show()

plt.figure()
plt.plot(fODE, rODE, 'b-', label='ODE')
    
plt.grid()
plt.legend(loc='best')
plt.xlabel('Foxes')    
plt.ylabel('Rabbits')
plt.title('Evolution of fox and rabbit populations')
    # f2.savefig('rabbits_and_foxes_2.png')
plt.show()

