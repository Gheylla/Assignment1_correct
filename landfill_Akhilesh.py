import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

# Definition of Rate Equations
def dydt(t, Y):
    """ Rate of change of storage in cover layer Scl and 
    storage in the waste layer Swb. """
    return np.array([J(t) - Lcl(t) - E(t), 
                     1 - B(t) * Lcl(t) - E(t)])

# Definition of parameters
S_Evmax = 1
S_Evmin = 0.25
Scl_min = 0.1
Scl_max = 1
Swb_max = 1 
Swb_min = 0.1
a = 0.1                         #saturated hydraulic conductivity [m/day]
bcl = 0.01                      # empirical parameter
bwb = 0.01
base_area = 28355               #[m2]
top_area = 9100                 #[m2]
slope_width = 38                #[m]
waste_body_height = 12          #[m]
cover_layer_height = 1.5        #[m]
waste = 281083000               #wet weight [kg]


def fscl():
    if Scl < S_Evmin:
        fscl = 0
    elif S_Evmin <= Scl <= S_Evmax:
        fscl = (Scl - S_Evmin) / (S_Evmax - S_Evmin)
    else fscl = 1

#Leaching rates
Lcl = a * (((Scl - Scl_min) / (Scl_max - Scl_min)) ^ bcl)
Lwb = a * (((Swb - Swb_min) / (Swb_max - Swb_min)) ^ bwb)

# Evaporation model
E(t) = pEv(t) * fcrop * fscl

# Total storage in a layer can never exceed the volume of the pore space


# Initial case
Y0 = np.array([0, 0])

# Definition of output times
tOut = np.linspace(0, 100, 200)              # time
nOut = np.shape(tOut)[0]
    
tic()

t_span = [tOut[0], tOut[-1]]
import scipy.integrate as spint
YODE = spint.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, method='RK45', vectorized=True, rtol=1e-5 )
    # infodict['message']                     # >>> 'Integration successful.'
rODE = YODE.y[0,:]
fODE = YODE.y[1,:]

toc()

# Plot results with matplotlib
plt.figure()
plt.plot(tOut, rODE, 'r-', label='RODE')
plt.plot(tOut, fODE, 'b-', label='FODE')

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
