import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import numpy as np

def graph_Likelihood(likelihood):
    """
        This function graphs the abs(likelihood) of the accepted values, in order
        to burn when necessary.

        Input:
        - likelihood (list): list with the likelihood of the accepted values.
    """
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(range(1,len(likelihood)+1), [abs(x) for x in likelihood])
    ax.set_xlabel("Accepted Values")
    ax.set_ylabel("Likelihood")
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
def getValues():
    """
        This function extracts all values from .txt after the c++ run.
    """

    f1 = open("tmpRes/accepted.txt", "r")
    f2 = open("tmpRes/likelihoodAccepted.txt", "r")
    f3 = open("tmpRes/rejected.txt", "r")

    likelihood = []
    accepted = []
    rejected = []

    for x, y in zip(f1.readlines(), f2.readlines()):
        accepted.append([float(ii) for ii in x.split(" ")[:-1]])
        likelihood.append(float(y))
    for x in f3.readlines():
        rejected.append([float(ii) for ii in x.split(" ")[:-1]])
    return likelihood, accepted, rejected

def burn_Result(result, index):
    """
        This function burns the result until the first index.

        Input:
        - result (list): list with the data to be burned
        - index (int>0): number until which the data will be burned
    """
    return result[index:]

def graph_Confidence(result, error, resolution=10):
    """
        This function graphs the confidence of all the parameters.

        Input:
        - result (list of lists): list with all the results after the calculation
        - error (list): list with the uncertaintities for the time delay
    """
    # H_0,omega_m0, omega_q0, alpha, alpha_x, m
    H_0 = [ii[0] for ii in result]
    omega_m0 = [ii[1] for ii in result]
    omega_q0 = [ii[2] for ii in result]
    alpha = [ii[3] for ii in result]
    alpha_x = [ii[4] for ii in result]
    m = [ii[5] for ii in result]
    newShape = [H_0, omega_m0, omega_q0, alpha, alpha_x, m]
    corr = [(0,1), (0,2), (0,3), (0,4), (0,5), (5,1), (5,2), (5,3), (5,4),
        (4,1),(4,2),(4,3),(3,1),(3,2),(2,1)]
    subplots = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,0), (1,1), (1,2), (1,3),
    (2,0), (2,1), (2,2), (3,0), (3,1), (4,0)]

    axisLabels = [(r"$H_0$", r"$\Omega_{m_0}$"), (r"$H_0$", r"$\Omega_{q_0}$"),
    (r"$H_0$", r"$\alpha$"), (r"$H_0$", r"$\alpha_x$"), (r"$H_0$", r"$m$"),
    (r"$m$",r"$\Omega_{m_0}$"), (r"$m$",r"$\Omega_{q_0}$"), (r"$m$",r"$\alpha$"),
    (r"$m$",r"$\alpha_x$"), (r"$\alpha_x$",r"$\Omega_{m_0}$"),
    (r"$\alpha_x$",r"$\Omega_{q_0}$"), (r"$\alpha_x$", r"$\alpha$"),
    (r"$\alpha$",r"$\Omega_{m_0}$"), (r"$\alpha$", r"$\Omega_{q_0}$"),
    (r"$\Omega_{q_0}$", r"$\Omega_{m_0}$")]

    f = plt.figure(figsize=(15,15))
    gs = GridSpec(5, 5)

    for ii in range(15):
        g = f.add_subplot(gs[subplots[ii][0], subplots[ii][1]])
        g.scatter(newShape[corr[ii][0]], newShape[corr[ii][1]], s=1, color="k")
        g.set_xlabel(axisLabels[ii][0])
        g.set_ylabel(axisLabels[ii][1])
        limits = [g.get_xlim(), g.get_ylim()]
        if limits[0][0]<0:
            g.set_xlim(0, limits[0][1])
        if limits[1][0]<0:
            g.set_ylim(0, limits[1][1])
        cov = np.cov(newShape[corr[ii][0]], newShape[corr[ii][1]])
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        #3 mean deviations
        for j in range(1, 4):
            ell = Ellipse(xy=(np.mean(newShape[corr[ii][0]]), np.mean(newShape[corr[ii][1]])),
                          width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                          angle=np.rad2deg(np.arccos(v[0, 0])))
            ell.set_facecolor('none')
            ell.set_edgecolor("k")
            g.add_artist(ell)
        print("%i/%i   "%(ii+1,15), end="\r")
        #A,B,C = info_Contourn(result[-1],limits, data, error, ii, N=resolution)
        #g.contour(A,B,C, 300, alpha=0.6)
        #g.scatter(newShape[corr[ii][0]], newShape[corr[ii][1]], s=1, color="k")
    print()
    plt.tight_layout()
    plt.savefig("result.pdf")


def likelihood_Function(H_0, omega_m0, omega_q0, alpha,
        alpha_x, m, data, error):
        """
            This function returns the likelihood function given the set of
            cosmological parameters and the experimental data.

            Input
            - H_0 (float): Hubble parameter
            - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters
            - data (list of lists): set of data from the observations

            Output:
            - L (float): likelihood function
        """
        #Unpacking
        z_ds = [x[0] for x in data]
        z_ss = [x[1] for x in data]
        delta_ts = [x[2] for x in data]
        theta_is = [x[3] for x in data]
        theta_js = [x[4] for x in data]


        acumulator = 0
        for x in range(len(data)):
            acumulator -= (delta_ts[x]-time_delay(theta_is[x], theta_js[x],
             H_0,z_ds[x], z_ss[x], omega_m0, omega_q0, alpha, alpha_x, m))**2/error[x]**2

        return acumulator
