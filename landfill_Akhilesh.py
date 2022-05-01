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
base_area = 28355
S_Evmin = 0
S_Evmax = 1

table_final['leacheate'] = table_final['leachate']/base_area

def dydt(t, Y):

    #Leaching rates

    Lcl = acl * (((Y[0] - Scl_min) / (Scl_max - Scl_min)) ^ bcl)
    Lwb = awb * (((Y[1] - Swb_min) / (Swb_max - Swb_min)) ^ bwb)
    
#reduction factor reducing evapotranspiration under dry soil conditions

    if Y[0] < S_Evmin:
        fred = 0
    elif S_Evmin <= Y[0] <= S_Evmax:
        fred = (Y[0] - S_Evmin) / (S_Evmax - S_Evmin)
    else: 
        fred = 1
    
## Total storage in a layer can never exceed the volume of the pore space##
    if Y[0] < Scl_min:
        Y[0] = Scl_min
    elif Y[0] > Scl_max:
            Y[0] = Scl_max   
    if Y[1] < Swb_min:
            Y[1] = Swb_min
    elif Y[1] > Swb_max:
            Y[1] = Swb_max 
            
    # Evaporation model
    E = evaporation[int(t)] * fcrop * fred
    
# B- beta term that allows a certain fraction of water leaching from the cover layer to directly enter the drainage layer
    B = beta0 * ((Y[0,:] - Scl_min) / (Scl_max - Scl_min)) 
        

# Definition of Rate Equations
    dydt = np.zeros(2)
    dydt[0, 1] = np.array(rain[int(t)] - Lcl - E, 1 - B * Lcl - Lwb)
    
    rODE = Y[0,:]
    fODE = Y[1,:]
    global rODE
    global fODE
    Q_cal = np.zeros(2757)
    E = np.zeros(2757)
    
    # Calculate the storage rate from the measured data
    
    Q_cal = beta0 * ((rODE - Scl_min) / (Scl_max - Scl_min)) * acl * ((rODE - Scl_min) / (Scl_max - Scl_min)) ** bcl + awb * ((fODE - Swb_min) / (Swb_max - Swb_min)) ** bwb
    Q_cal = Q_cal * 28355   # Multiply the area
    print(Q_cal)
    Solution = spint.solve_ivp(dydt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
    return dydt


# Initial case
Y0 = np.array([0.8, 0.8])

# Definition of output times
tOut = np.linspace(0, len(evaporation), len(evaporation))              # time
nOut = np.shape(tOut)[0]
t_span = [tOut[0], tOut[-1]]

Q_rate = np.zeros(2757)
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

