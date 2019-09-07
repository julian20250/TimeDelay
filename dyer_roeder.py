#Numeric integration Dyer-Roeder Equation
# Try alpha=1 with alpha_1 func
import matplotlib.pyplot as plt
from mpmath import hyp2f1
#from scipy.special import hyp2f1
import numpy as np
from scipy.special import lpmn
#Constants
#Test [0.3,.5,1,1,4]
#Flat Universe [1,0,0,0, 2]
sigma_m0, sigma_q0, alpha, alpha_x, m = [.7,0,1,0,0]
beta= ((25-24*alpha)**.5)/4
a, b=5/4+beta, 5/4-beta
c=1/2
#Starting point
z_0=0
z_1=0
z_2=5
# Integration parameters
h=0.001 #Length of steps
N=int(z_2/h) #Number of steps

def f_1(z):
    """
    This function multiplies the second order derivative in the
    Dyer-Roeder equation.
    """
    return sigma_m0*z+1+sigma_q0*(1+z)**(m-2)*(1-1/(1+z)**(m-2))

def f_2(z):
    """
    This function multiplies the first order derivative in the
    Dyer-Roeder equation.
    """
    return (7*sigma_m0*z/2+sigma_m0/2+3+sigma_q0*(1+z)**(m-2)*\
            ((m+4)/2-3/(1+z)**(m-2)))/(1+z)
def f_3(z):
    """
    This function multiplies the function r(z) in the
    Dyer-Roeder equation.
    """
    return (3/(2*(1+z)**4))*(alpha*sigma_m0*(1+z)**3+alpha_x*m/3*\
            sigma_q0*(1+z)**m)
def F(z,r,v):
    """
    This function is obtained clearing the second derivative of r(z)
    in the Dyer-Roeder equation.
    """
    return -(f_2(z)*v+f_3(z)*r)/f_1(z)

def flat_universe(z):
    r_list = [ ]
    for ii in z:
        r_list.append(2*(1-1/(1+ii)**(5/2.))/5)
    return r_list

def beta_func(z):
    r_list = [ ]
    for ii in z:
        r_list.append(((1+ii)**beta-(1+ii)**(-beta))/(2*beta*(1+ii)**(5/4)))
    return r_list
def hypergeometric(z):
    #u_3^((2))(x)	=	(-z)^(b-c)(1-z)^(c-a-b)_2F_1(1-b,c-b;a+1-b;z^(-1))
    #u_6^((4))(x)	=	z^(b-c)(1-z)^(c-a-b)_2F_1(c-b,1-b;c+1-a-b;1-z^(-1))
    #u_2^((3))(x)	=	z^(-a)_2F_1(a,a+1-c;a+b+1-c;1-z^(-1))
    #u_3(x)	=	z^(-a)_2F_1(a,a+1-c;a+1-b;z^(-1))
    #ii**(-a)*hyp2f1(a,1+a-c,a+1-b,ii**(-1))

    #u_4(x)	=	z^(-b)_2F_1(b+1-c,b;b+1-a;z^(-1))
    x=[]
    for ii in z:
        x.append(((1+sigma_m0*ii)/(1-sigma_m0)))
    #print(x)
    r_list=[]
    for ii in x:
        r_list.append(hyp2f1(a,b,c,ii**(-1)))
        #print(ii**(b-c)*(1-ii)**(c-a-b))
    #print(x)
    return r_list
def legendre(z):
    x=[]
    for ii in z:
        x.append(((1+sigma_m0*ii)/(1-sigma_m0))**.5)
    r_list=[]
    for ii in x:
        #print(lpmn((beta-1)/2, 2,ii)[0][-1][-1])
        r_list.append(lpmn((4*beta-1)/2, 2,ii)[0][-1][-1]/(ii**2-1))
    return r_list
def alpha_1(z):
    r_list=[]
    for ii in z:
        r_list.append(2*(sigma_m0*ii-(2-sigma_m0)*((sigma_m0*ii+1)**.5-1))/(sigma_m0**2*(1+ii)**2))
    return r_list
#Initial conditions
r_0 = 0
#v_0 = 1/((1+z_0)**2*(sigma_m0*z_0+1+sigma_q0*(1+z_0)**(m-2)*\
#        (1-1/(1+z_0**(m-2))))**.5)
v_0 = 1/((1+z_0)**2*(sigma_m0*z_0+1+sigma_q0*(1+z_0)**(m-2)*\
        (1-1/(1+z_0)**(m-2)))**.5)

#Numeric Integration Zone
r = [r_0]
v = [v_0]
z = z_0
z_array = [z_0]
for ii in range(N):
    #Runge Kutta Fourth Order
    r.append(r[-1]+h*v[-1])

    k1 = F(z, r[-1], v[-1])
    k2 = F(z+h/2, r[-1], v[-1]+k1*h/2)
    k3 = F(z+h/2, r[-1], v[-1]+k2*h/2)
    k4 = F(z+h, r[-1], v[-1]+k3*h)

    v.append(v[-1]+h*(k1+2*k2+2*k3+k4)/6)
    z += h
    z_array.append(z)

string = r"$\Omega_{m_0} = %1.2f$"%sigma_m0+"\n"+r"$\Omega_{Q_0} = %1.2f$"%sigma_q0+"\n"+r"$\overline{\alpha} = %1.2f$"%alpha+"\n"+r"$\overline{\alpha_x} = %1.2f$"%alpha_x+"\n"+r"$m=%1.2f$"%m+"\n"+r"$z_0 = %1.2f$"%z_0

#Graphic Zone
plt.plot(z_array, r, label="Numeric Integration")
#plt.plot(z_array, (1+sigma_m0*np.array(z_array))/(1-sigma_m0))
#print(hypergeometric(z_array))
plt.plot(z_array, alpha_1(z_array), "--",label=r"$r(z)=F(a,b,c,x(z))$")
"""
List:
r(z)=\frac{2}{5}\left(1-\frac{1}{(1+z)^{5/2}} \right)
r(z)=\frac{(1+z)^\beta-(1+z)^{-\beta}}{2\beta(1+z)^{5/4}}
r(z)=F(a,b,c,x(z))
r(z)=\frac{2}{\Omega^{2}(1+z)^{2}}[\Omega z-(2-\Omega)(\sqrt{\Omega z+1}-1)]
"""
plt.xlabel(r"$z$")
plt.ylabel(r"$r(z)$")
plt.text(0.7, 0.60, string, fontsize=10, transform=plt.gcf().transFigure,
bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
plt.legend()
plt.xlim(z_1,z_2)
plt.tight_layout()
plt.show()
