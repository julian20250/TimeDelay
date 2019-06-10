import sympy as sp
from astropy import units as u
import sympy.printing as printing
from sympy import init_session
from scipy.constants import G
init_session(quiet=True)

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
m=[0.33e24, 4.87e24, 5.97e24, 0.642e24, 1898e24, 568e24, 86.8e24, 102e24
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
gamma_value=[G*M*x for x in m]

#Precession (1''/100yrs)
omega_obs =[43.1, 8, 5, 1.3624, 0.07, 0.014]
omega_uncertaint =[0.5, 5, 1, 5e-4, 4e-3, 2e-3]

#Real Integration ================================
total=6 #Planets taken
for x in range(1):
    delta_val=np.linspace(0, 0.2)

    #Drawing Zone
    f=plt.figure()
    plt.ylabel("Precesión (''/siglo)")
    plt.xlabel("$\Delta$")
    plt.title(names[x])
    plt.tight_layout()
    plt.savefig("integralResults/%s"%names[x])
    print("End %s"$names[x])
