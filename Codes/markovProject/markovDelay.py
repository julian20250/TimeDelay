#Database: https://www.cfa.harvard.edu/castles/
#Interesting References:
#https://towardsdatascience.com/from-scratch-bayesian-inference-markov-chain-monte-carlo-and-metropolis-hastings-in-python-ef21a29e25a
#https://github.com/DanielTakeshi/MCMC_and_Dynamics/blob/master/standard_mcmc/Quick_MH_Test_Example.ipynb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
c =299792.458 #km/s
kmToMPc = 3.240779289666e-20

def r_dyer_roeder(z, omega_m0, omega_q0, alpha, alpha_x, m, h=0.001):
    """
        This function returns the value for r(z) given a set of cosmological
        parameters (\Omega_{m_0}, \Omega_{q_0}, \alpha, \alpha_x, m) solving
        numerically the Dyer-Roeder equation.

        Input:
        - z (float): redshift
        - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters
        - h (float): Length of integration steps

        Output:
        - r(z) (float)
    """
    def f_1(z):
        """
        This subfunction multiplies the second order derivative in the
        Dyer-Roeder equation.
        """
        return omega_m0*z+1+omega_q0*(1+z)**(m-2)*(1-1/(1+z)**(m-2))

    def f_2(z):
        """
        This subfunction multiplies the first order derivative in the
        Dyer-Roeder equation.
        """
        return (7*omega_m0*z/2+omega_m0/2+3+omega_q0*(1+z)**(m-2)*\
                ((m+4)/2-3/(1+z)**(m-2)))/(1+z)
    def f_3(z):
        """
        This subfunction multiplies the function r(z) in the
        Dyer-Roeder equation.
        """
        return (3/(2*(1+z)**4))*(alpha*omega_m0*(1+z)**3+alpha_x*m/3*\
                omega_q0*(1+z)**m)

    def F(z,r,v):
        """
        This subfunction is obtained clearing the second derivative of r(z)
        in the Dyer-Roeder equation.
        """
        return -(f_2(z)*v+f_3(z)*r)/f_1(z)

    N=int(z/h) #Number of steps

    z_0 = 0
    #Initial conditions
    r_0 = 0
    v_0 = 1/((1+z_0)**2*(omega_m0*z_0+1+omega_q0*(1+z_0)**(m-2)*\
            (1-1/(1+z_0)**(m-2)))**.5)

    #Numeric Integration Zone
    r = r_0
    v = v_0
    z = z_0

    for ii in range(N):
        #Runge Kutta Fourth Order
        r = r+h*v
        k1 = F(z, r, v)
        k2 = F(z+h/2, r, v+k1*h/2)
        k3 = F(z+h/2, r, v+k2*h/2)
        k4 = F(z+h, r, v+k3*h)
        v=v+h*(k1+2*k2+2*k3+k4)/6
        z += h
    return r

def cosmological_Distances(H_0,z_a, z_b, omega_m0, omega_q0, alpha, alpha_x, m):
    """
        The output of this function is the cosmological distance in function
        of the cosmological parameters, the redshift and the Hubble parameter.

        Input
        - H_0 (float): Hubble parameter
        - z_a,z_b (float): redshift
        - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters

        Output:
        - D (float): Cosmological distance
    """
    return c*r_dyer_roeder(abs(z_a-z_b), omega_m0, omega_q0, alpha, alpha_x, m)/H_0

def time_delay(th_1, th_2, H_0,z_d, z_s, omega_m0, omega_q0, alpha, alpha_x, m):
    """
        This function calculates the time delay, given the redshift of the source
        z_s, the redshift of the deflector z_d, the cosmological parameters and
        the angular positions of these two objects.

        Input
        - th_1, th_2 (float): angular positions
        - H_0 (float): Hubble parameter
        - z_s,z_d (float): redshift
        - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters

        Output:
        - Dt (float): time delay
    """
    D_d = cosmological_Distances(H_0,z_d, 0, omega_m0, omega_q0, alpha, alpha_x, m)
    D_ds = cosmological_Distances(H_0,z_d, z_s, omega_m0, omega_q0, alpha, alpha_x, m)
    D_s = cosmological_Distances(H_0,z_s, 0, omega_m0, omega_q0, alpha, alpha_x, m)

    DTheta_squared = abs(th_1**2-th_2**2)
    return (1+z_d)*D_d*D_s*DTheta_squared/(2*D_ds*c*kmToMPc)

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
            acumulator -= (delta_ts[x]-time_delay(theta_is[x], theta_js[x], H_0,z_ds[x], z_ss[x], omega_m0, omega_q0, alpha, alpha_x, m))**2/error[x]**2

        return acumulator

def model_Prior(H_0,omega_m0, omega_q0, alpha, alpha_x, m, deviations):
    """
        This prior distribution discards all the values that doesn't have
        physical meaning.

        Input
        - H_0 (float): Hubble parameter
        - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters
        - means (list): ordered list with the means of all the parameters
        - deviations (list): ordered list with the deviations of all the parameters

        Output:
        - Prior value (float): value of the prior function
    """
    parameters = [H_0,omega_m0, omega_q0, alpha, alpha_x, m]
    #Accept only valid cosmological parameters
    cond = H_0<0 or omega_m0>1 or omega_q0>1 or omega_m0<0 or omega_q0<0\
            or alpha<0 or alpha_x<0 or m<0 or (omega_m0+omega_q0)>1

    #for ii in deviations:
    #    cond = cond or (ii<0)
    if cond:
        return 0
    return 1

def transition_Model(means, deviations):
    """
        This function chooses the new values to be compared with the last values
        using a normal like distribution for all the parameters.

        Input:
        - means (list): list with all the means of the cosmological parameters
        - deviations (list): list with all the deviations of the cosmological parameters

        Output:
        - l (list): list with the new means and deviations for the cosmological parameters,
        structured as follows: new_means+new_deviations
    """
    l = []
    #hyperDeviations=[0.5, 0.05,0.05,.5,.5,.5]
    for x,y in zip(means, deviations):
        l.append(x+np.random.normal(0,y))
    #for x,y in zip(deviations,hyperDeviations):
    #    l.append(np.random.normal(x,y))
    for x in deviations:
        l.append(x)
    return l


def acceptance_rule(old, new):
    """
        This function determines if accept or reject the new data, based on the
        detailed balance method.

        Input:
        - old (float): old value of detailed balance condition
        - new (float): new value of detailed balance condition

        Output:
        - (Bool): True if accepting, False if rejecting
    """
    if new>old:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (accept < np.exp(new-old))

def metropolis_Hastings(param_init, iterations, deviations,data, error, accept_limit=-1,
                        decreaseDeviation=-1):
    """
        This function implements the Metropolis Hastings method for obtaining
        the best possible parameters from a set of data.

        Input:
        - param_init (list): initial parameters
        - iterations (int): number of iterations
        - deviations (list): initial deviations
        - data (list of lists): experimental data
        - accept_limit (int>0, optional): stop when reaching len(accepted_values)==accept_limit
        - decreaseDeviation (int>0, optional) : decrease Deviations when not accepting
        after decreaseDeviation iterations

        Output:
        - [accepted, rejected,likely_accepted] (list of lists).
    """
    #Order of Data
    # H_0,omega_m0, omega_q0, alpha, alpha_x, m

    x = param_init
    accepted = []
    rejected = []
    likely_accepted = []
    x = x+deviations
    trueTime = 0
    timesWithoutAccept=0
    if accept_limit==-1:
        for ii in range(iterations):
            x_new =  transition_Model(x[:6], x[6:])

            x_lik = likelihood_Function(x[0], x[1], x[2], x[3],
                    x[4], x[5], data, error)
            x_new_lik = likelihood_Function(x_new[0], x_new[1], x_new[2], x_new[3],
             x_new[4], x_new[5],data, error)

            if (acceptance_rule(x_lik + np.log(model_Prior(x[0], x[1], x[2], x[3],
                    x[4], x[5], x[6:])),x_new_lik+np.log(model_Prior(x_new[0], x_new[1], x_new[2], x_new[3],
                     x_new[4], x_new[5], x_new[6:])))):
                x = x_new
                accepted.append(x_new)
                likely_accepted.append(x_new_lik)
                timesWithoutAccept = 0
                trueTime = 0
            else:
                rejected.append(x_new)
                timesWithoutAccept += 1
                trueTime +=1

            if decreaseDeviation !=-1:
                if timesWithoutAccept > decreaseDeviation:
                    x[6:] = [jj/1.2 for jj in x[6:]]
                    timesWithoutAccept = 0
            print("Iteration %i/%i. Accepted Values: %i. Iterations without accepting: %i      "%(ii+1,iterations,len(accepted), trueTime), end="\r")
    else:
        ii=1
        while(len(accepted)<accept_limit):
            x_new =  transition_Model(x[:6], x[6:])

            x_lik = likelihood_Function(x[0], x[1], x[2], x[3],
                    x[4], x[5], data, error)
            x_new_lik = likelihood_Function(x_new[0], x_new[1], x_new[2], x_new[3],
             x_new[4], x_new[5],data, error)

            if (acceptance_rule(x_lik + np.log(model_Prior(x[0], x[1], x[2], x[3],
                    x[4], x[5], x[6:])),x_new_lik+np.log(model_Prior(x_new[0], x_new[1], x_new[2], x_new[3],
                     x_new[4], x_new[5], x_new[6:])))):
                x = x_new
                accepted.append(x_new)
                likely_accepted.append(x_new_lik)
                timesWithoutAccept = 0
                trueTime = 0
            else:
                rejected.append(x_new)
                timesWithoutAccept += 1
                trueTime +=1
            if decreaseDeviation !=-1:
                if timesWithoutAccept > decreaseDeviation:
                    x[6:] = [jj/1.2 for jj in x[6:]]
                    timesWithoutAccept = 0

            print("Iteration %i. Accepted Values: %i/%i. Iterations without accepting: %i      "%(ii+1,len(accepted), accept_limit, trueTime), end="\r")
            ii+=1

    print()
    return [accepted, rejected,likely_accepted]

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

def graph_Confidence(result, data, error, resolution=10):
    """
        This function graphs the confidence of all the parameters.

        Input:
        - result (list of lists): list with all the results after the calculation
        - data (list of lists): list with all the experimental values
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
        g.scatter(newShape[corr[ii][0]], newShape[corr[ii][1]], s=1, color="white")
        g.set_xlabel(axisLabels[ii][0])
        g.set_ylabel(axisLabels[ii][1])
        limits = [g.get_xlim(), g.get_ylim()]
        A,B,C = info_Contourn(result[-1],limits, data, error, ii, N=resolution)
        g.contour(A,B,C, 300, alpha=0.6)
        g.scatter(newShape[corr[ii][0]], newShape[corr[ii][1]], s=1, color="k")

    plt.tight_layout()
    plt.savefig("result.pdf")

def burn_Result(result, index):
    """
        This function burns the result until the first index.

        Input:
        - result (list): list with the data to be burned
        - index (int>0): number until which the data will be burned
    """
    return result[index:]
def info_Contourn(lastResult,limits, data, error, index, N=10):
    """
        This function makes the arrays to graph the confidence regions.

        Input:
        - lastResult (list): list with the last data obtained.
        - limits (list of lists): list with all the limits of the graphics
        - data (list of lists): list with all the experimental data
        - error (list): list with the uncertaintities of the time delay
        - index (int): number that determines the graphic to be done

        Output:
        - (tuple): matrices to be contoured
    """

    corr = [(0,1), (0,2), (0,3), (0,4), (0,5), (5,1), (5,2), (5,3), (5,4),
    (4,1),(4,2),(4,3),(3,1),(3,2),(2,1)]
    #Avoid negative limits
    x_low, y_low = limits[0][0], limits[1][0]
    if limits[0][0]<0:
        x_low = 0
    if limits[1][0]<0:
        y_low = 0
    x_arr = np.linspace(x_low, limits[0][1], N)
    y_arr = np.linspace(y_low, limits[1][1], N)

    X_a,Y_a = np.meshgrid(x_arr, y_arr)
    Z_a = np.zeros((N,N))
    count = 1
    for ii, x in enumerate(x_arr):
        for jj,y in enumerate(y_arr):
            tmpResult = lastResult.copy()
            tmpResult[corr[index][0]] = x
            tmpResult[corr[index][1]] = y
            Z_a[ii,jj]=likelihood_Function(tmpResult[0], tmpResult[1],
             tmpResult[2],tmpResult[3], tmpResult[4], tmpResult[5], data, error)
            print("Graphic %i/15. Block %i/%i  "%(index+1, count, N*N), end="\r")
            count+=1

    return X_a, Y_a, Z_a
