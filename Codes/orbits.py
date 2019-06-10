import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import G
case=int(input("Case 1,2 > "))
#Constants
M=1.989e30
m=5.972e24
lamb=7.5e6

# In order to make eccentricity equal to 0.5
gamma=G*M*m
mu=m*M/(m+M)
L=(0.375*gamma**2*mu/3)**(1/3.)
E=-3*L

delta=np.array([-.6,-.1,0,.1,.6])
beta_0=mu*gamma/L**2
beta_1=2*mu*E/L**2-2*mu*gamma*delta/(lamb*L**2*(1+delta))
l=1./beta_0
if case==1:
    epsilon=np.sqrt(1+l**2*beta_1)
if case==2:
    beta_2=mu*gamma*delta/(L**2*lamb**2*(1+delta))
    epsilon=np.sqrt(1+l**2*beta_1+l**3*beta_2c)
#First Graph
count=0
varphi=np.linspace(0,2*np.pi, 1000)
styles=['-.', ':', '-', '--','-.']
e_N=epsilon[2]
r_N=l/(1+e_N*np.cos(varphi))

fig, axes = plt.subplots(2, 1)
for x in delta:
    r=l/(1+epsilon[count]*np.cos(varphi))
    axes[0].plot(r*np.cos(varphi), r*np.sin(varphi),
    label="$\delta=%.1f$"%x, linestyle=styles[count])
    d=abs(r-r_N)
    if count==0:
        r_max=max(d)
        axes[1].scatter(varphi[d.argmax()], r_max)
        axes[1].text(varphi[d.argmax()], r_max-500000, "%f m"%r_max)
    if count!=2:
        axes[1].plot(varphi, d, label="$\delta=%.1f$"%x,
         linestyle=styles[count])
    count+=1
axes[0].axhline(0, color='black')
axes[0].axvline(0, color='black')
axes[0].set_xlabel("x (meters)")
axes[0].set_ylabel("y (meters)")
axes[0].legend()

axes[1].set_xlabel(r"$\theta$")
axes[1].set_ylabel(r"$|r_N(\theta)-r(\theta,\delta)|$")
axes[1].legend()
plt.tight_layout()
plt.show()
