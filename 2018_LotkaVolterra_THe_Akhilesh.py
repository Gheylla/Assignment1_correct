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

# Definition of Rate Equation
def dYdt(t, Y):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a*Y[0] - b*Y[0]*Y[1],
                     -c*Y[1] + d*Y[0]*Y[1]])

# Definition of parameters
a = 1
b = 0.25
c = 0.1
d = 0.01

# Initial case, 10 rabbits, 5 foxes
Y0 = np.array([10, 5])

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