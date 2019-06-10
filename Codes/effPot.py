import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition)



#Constants
G=6.6738e-11
lamb=1.6e16
M=1e24

r_lamb=np.linspace(0.1, 0, 20000, endpoint=False)
delta_val=[-0.6,-0.1,0,0.1,0.6]
#delta_val=[x*100 for x in delta_val]
Phi=[(2e-4/r_lamb**2-1/((1+x)*r_lamb)-x*np.exp(-r_lamb)/((1+x)*r_lamb))/1000 for x in delta_val]
#maxim=min([min(x) for x in Phi])
#Phi=[-x/maxim for x in Phi]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
styles=['-.', ':', '-', '--','-.']
for x in range(len(Phi)):
    print(Phi[x][2])
    ax.plot(r_lamb, Phi[x], label="$\delta=%.1f$"%delta_val[x],
     linestyle=styles[x])

axins = plt.axes([5,5,1,1])
ip = InsetPosition(ax, [0.4,0.08,0.5,0.5])
axins.set_axes_locator(ip)
for x in range(len(Phi)):
    axins.plot(r_lamb, Phi[x], label="$\delta=%.1f$"%delta_val[x],
    linestyle=styles[x])

x1, x2, y1, y2 = 3.8e-4, 4.3e-4, -1.252, -1.247
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.set_xlim([-0.0005, 0.05])
ax.set_ylim([-1.5, 1])
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
ax.set_xlabel('$r/\lambda$')
ax.set_ylabel('$V_{eff}(r/\lambda)$')
ax.legend()
plt.tight_layout()
plt.draw()
plt.show()
