#Database: https://www.cfa.harvard.edu/castles/
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
        return sigma_m0*z+1+sigma_q0*(1+z)**(m-2)*(1-1/(1+z)**(m-2))

    def f_2(z):
        """
        This subfunction multiplies the first order derivative in the
        Dyer-Roeder equation.
        """
        return (7*sigma_m0*z/2+sigma_m0/2+3+sigma_q0*(1+z)**(m-2)*\
                ((m+4)/2-3/(1+z)**(m-2)))/(1+z)
    def f_3(z):
        """
        This subfunction multiplies the function r(z) in the
        Dyer-Roeder equation.
        """
        return (3/(2*(1+z)**4))*(alpha*sigma_m0*(1+z)**3+alpha_x*m/3*\
                sigma_q0*(1+z)**m)

    def F(z,r,v):
        """
        This subfunction is obtained clearing the second derivative of r(z)
        in the Dyer-Roeder equation.
        """
        return -(f_2(z)*v+f_3(z)*r)/f_1(z)

    N=int(z_2/h) #Number of steps
    #Initial conditions
    r_0 = 0
    v_0 = 1/((1+z_0)**2*(sigma_m0*z_0+1+sigma_q0*(1+z_0)**(m-2)*\
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
        v=v+h*(k1+2*k2+2*k3+k4)/6)
        z += h
    return r

def cosmological_Distances(H_0,z_a, z_b, omega_m0, omega_q0, alpha, alpha_x):
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
    return r_dyer_roeder(abs(z_a-z_b), omega_m0, omega_q0, alpha, alpha_x, m)/H_0

def time_delay(th_1, th_2, H_0,z_d, z_s, omega_m0, omega_q0, alpha, alpha_x):
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
    D_d = cosmological_Distances(H_0,z_d, 0, omega_m0, omega_q0, alpha, alpha_x)
    D_ds = cosmological_Distances(H_0,z_d, z_s, omega_m0, omega_q0, alpha, alpha_x)
    D_s = cosmological_Distances(H_0,z_s, 0, omega_m0, omega_q0, alpha, alpha_x)

    DTheta_squared = abs(th_1**2-th_2**2)
    
    return (1+z_d)*D_d*D_s*DTheta_squared/(2*D_ds)
