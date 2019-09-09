import sympy as sp
from astropy import units as u
from sympy import init_session
import numpy as np
from scipy.integrate import quad
from scipy.constants import G as Grav
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#About Integration ================================
mu, gamma, delta, lamb, eta = sp.symbols("mu gamma delta lambda eta")
u0, u1, v= sp.symbols("u_0 u_1 v")

def intFun(delta_value, fun, approx):
    fun=fun.replace(delta, delta_value)
    fun=sp.lambdify(v, fun, "numpy")
    return approx-quad(fun, 0,1)[0]

to_integrate = 1/(2-1/(lamb**2*(1+delta)*u0*(u0+eta)))*((-2-eta/u0+2*v+eta*v**2/u0-2*u0*v/\
(u0+eta)-eta*v**2/(u0+eta)+2*u0/(u0+eta*v)+eta/(u0+eta*v))/(lamb**2*(1+delta))+2*eta**2*v*(1-v))

to_integrate=(to_integrate)**(-1/2.)
#=======================================
#Constants ..............................
lamb_value = (5000*u.au).to(u.meter).value #Compton Wavelength (in meters)
M=1.989e30 #Sun Mass
m=[0.33e24, 4.87e24, 5.97e24, 0.642e24, 1898e24, 568e24, 86.8e24, 102e24,
    0.0146e24] #Planets' mass
names =["Mercurio", "Venus", "Tierra", "Marte", "Júpiter", "Saturno",
    "Urano", "Neptuno", "Plutón"] #Planets' names

a=[0.39, 0.72, 1, 1.52, 5.20, 9.54, 19.2, 30.1, 39.4] #Semimajor axis (ua)
a=[(x*u.au).to(u.meter).value for x in a] #Semimajor axis (meters)
epsilon=[0.206, 0.007, 0.017, 0.093, 0.048, 0.056, 0.046, 0.009, 0.25] #Eccentricity

#Perihelion and Afelium
u0_value=[1./(x*(1-y)) for x,y in zip(a,epsilon)]
u1_value=[1./(x*(1+y)) for x,y in zip(a,epsilon)]

#Value of mu (reduced mass)
mu_value=[x*M/(x+M) for x in m]

#Gamma Value (GMm)
gamma_value=[Grav*M*x for x in m]

#Precession (1''/100yrs)
omega_obs =[43.1, 8, 5, 1.3624, 0.07, 0.014]
omega_uncertaint =[0.5, 5, 1, 5e-4, 4e-3, 2e-3]

#Period of orbit in earth's years
T=[0.24, 0.62, 1, 1.88, 11.86, 29.46, 84.1, 164.8, 247.7]

omega_inradians=[(omega_obs[x]*T[x]*np.pi/100/3600/180+2*np.pi)/2 for x in range(len(omega_obs))]

total=6
for x in range(total):
    function = (eta*to_integrate).replace(mu, mu_value[x]).replace(gamma,\
    gamma_value[x]).replace(lamb, lamb_value).replace(\
    eta, u1-u0).replace(u0, u0_value[x]).replace(u1, u1_value[x])

    print(fsolve(intFun, 0.1, args=(function, omega_inradians[x])))
