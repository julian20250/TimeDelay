import numpy as np
import matplotlib.pyplot as plt

H_0, like = np.loadtxt("tmpRes/likelihood.txt", unpack=True)
like = abs(like)
plt.plot(H_0, like)
plt.xlabel(r"$H_0$")
plt.ylabel("Likelihood")
plt.show()
