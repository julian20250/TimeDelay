import sympy as sp
from astropy import units as u
from sympy import init_session
import numpy as np
from scipy.integrate import quad
from scipy.constants import G as Grav
import matplotlib.pyplot as plt

#About Integration ================================
mu, gamma, delta, lamb, eta = sp.symbols("mu gamma delta lambda eta")
u0, u1, v= sp.symbols("u_0 u_1 v")

def g(x):
    return -2*mu*gamma*x-mu*gamma*delta/(lamb**2*(1+delta)*x)

def G(a,b,x):
    return (g(a)*(b**2-x**2)+g(b)*(x**2-a**2)-(b**2-a**2)*g(x))/\
    (g(a)-g(b))

def other_g(a,b,x):
    return G(a,b,x)**(-1/2.)

to_integrate=other_g(u0,u0+eta,u0+eta*v)
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

#Real Integration ================================
total=6 #Planets taken
for x in range(1):
    delta_val=np.linspace(0, 0.02)
    prec=[]
    for y in delta_val:
        function =(eta*to_integrate).replace(mu, mu_value[x]).replace(gamma,\
        gamma_value[x]).replace(delta, y).replace(lamb, lamb_value).replace(\
        eta, u1-u0).replace(u0, u0_value[x]).replace(u1, u1_value[x])

        int_fun=sp.lambdify(v, function, "numpy")
        prec.append(quad(int_fun, 0,1)[0])
        #print(prec[-1])
    prec=[(2*abs(z)-2*np.pi)*3600*180/np.pi for z in prec] #Precession in seconds
    prec=[z*100./T[x] for z in prec] #Precession in seconds per century
    print(prec)
    #Drawing Zone
    f=plt.figure()
    #Observacional
    plt.plot(delta_val, [omega_obs[x]-omega_uncertaint[x]]*len(delta_val), "b",
    label="Observaciones")
    plt.plot(delta_val, [omega_obs[x]-omega_uncertaint[x]]*len(delta_val), "b")

    #Teórico
    plt.plot(delta_val, prec, label="Precesión de Yukawa")
    #Belleza
    plt.ylabel("Precesión (''/siglo)")
    plt.xlabel("$\delta$")
    plt.title(names[x])
    plt.tight_layout()
    plt.legend()
    plt.savefig("integralResults/%s"%names[x])
    print("End %s"%names[x])
