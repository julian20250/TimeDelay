#include <math.h>
#include <random>
#include <stdio.h>
#include <fstream>
#include <chrono>
using namespace std;

const double c = 299792.458; //km/s
const double kmToMPc = 3.240779289666e-20;

double f_1(double z, double omega_m0, double omega_q0, double alpha,
   double alpha_x, double m){
     // This function multiplies the second order derivative in the
     //     Dyer-Roeder equation.
     return omega_m0*z+1+omega_q0*pow((1+z),(m-2))*(1-1/pow((1+z),(m-2)));
   }

double f_2(double z, double omega_m0, double omega_q0, double alpha,
   double alpha_x, double m){
     // This function multiplies the first order derivative in the
     //     Dyer-Roeder equation.
     return (7*omega_m0*z/2+omega_m0/2+3+omega_q0*pow((1+z),(m-2))*
     ((m+4)/2-3/pow((1+z),(m-2))))/(1+z);
   }

double f_3(double z, double omega_m0, double omega_q0, double alpha,
   double alpha_x, double m){
     // This function multiplies the function r(z) in the
     //     Dyer-Roeder equation.
     return (3/(2*pow((1+z),4)))*(alpha*omega_m0*pow((1+z),3)+alpha_x*m/3*
                 omega_q0*pow((1+z),m));
   }

double F(double z, double r, double v, double omega_m0, double omega_q0,
   double alpha, double alpha_x, double m){
     // This subfunction is obtained clearing the second derivative of r(z)
     // in the Dyer-Roeder equation.
     return -(f_2(z, omega_m0, omega_q0, alpha, alpha_x, m)*v+
     f_3(z, omega_m0, omega_q0, alpha, alpha_x, m)*r)/
     f_1(z, omega_m0, omega_q0, alpha, alpha_x, m);
}

double r_dyer_roeder(double z_0, double z_1, double omega_m0, double omega_q0,
  double alpha, double alpha_x, double m, double h=0.001){
    /*
      This function returns the value for r(z) given a set of cosmological
      parameters (\Omega_{m_0}, \Omega_{q_0}, \alpha, \alpha_x, m) solving
      numerically the Dyer-Roeder equation.

      Input:
      - z (doubles): redshift
      - (omega_m0, omega_q0, alpha, alpha_x, m) (doubles): cosmological parameters
      - h (float): Length of integration steps

      Output:
      - r(z) (float)
      */
    double z = z_0;
    int N = (z_1-z_0)/h;

    //Initial conditions
    double r_0 = 0;
    double v_0 = 1/(pow((1+z_0),2)*pow((omega_m0*z_0+1+omega_q0*pow((1+z_0),(m-2))*
    (1-1/pow((1+z_0),(m-2)))),.5));

    //Numeric integration zone
    double r = r_0;
    double v = v_0;

    for (int ii=0; ii<N; ii++){
      //Runge Kutta Fourth Order
          r = r+h*v;
          double k1 = F(z, r, v, omega_m0, omega_q0, alpha, alpha_x, m);
          double k2 = F(z+h/2, r, v+k1*h/2, omega_m0, omega_q0, alpha, alpha_x, m);
          double k3 = F(z+h/2, r, v+k2*h/2, omega_m0, omega_q0, alpha, alpha_x, m);
          double k4 = F(z+h, r, v+k3*h, omega_m0, omega_q0, alpha, alpha_x, m);
          v=v+h*(k1+2*k2+2*k3+k4)/6;
          z += h;
    }
    return r;
}

double cosmological_Distances(double H_0, double z_a, double z_b, double omega_m0,
   double omega_q0, double alpha, double alpha_x, double m){
     /*
    The output of this function is the cosmological distance in function
    of the cosmological parameters, the redshift and the Hubble parameter.

    Input
    - H_0 (float): Hubble parameter
    - z_a,z_b (float): redshift
    - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters

    Output:
    - D (double): Cosmological distance
    */
    return c*r_dyer_roeder(z_a,z_b, omega_m0, omega_q0, alpha, alpha_x, m)/H_0;
}

double time_delay(double th_1, double th_2, double H_0, double z_d, double z_s,
   double omega_m0, double omega_q0, double alpha, double alpha_x, double m){
    /*
        This function calculates the time delay, given the redshift of the source
        z_s, the redshift of the deflector z_d, the cosmological parameters and
        the angular positions of these two objects.

        Input
        - th_1, th_2 (double): angular positions
        - H_0 (double): Hubble parameter
        - z_s,z_d (double): redshift
        - (omega_m0, omega_q0, alpha, alpha_x, m) (doubles): cosmological parameters

        Output:
        - Dt (double): time delay
    */
    double D_d = cosmological_Distances(H_0,0, z_d, omega_m0, omega_q0, alpha, alpha_x, m);
    double D_ds = cosmological_Distances(H_0,z_d, z_s, omega_m0, omega_q0, alpha, alpha_x, m);
    double D_s = cosmological_Distances(H_0,0, z_s, omega_m0, omega_q0, alpha, alpha_x, m);

    double DTheta_squared = fabs(pow(th_1,2)-pow(th_2,2));
    return (1+z_d)*D_d*D_s*DTheta_squared/(2*D_ds*c*kmToMPc);
}

double likelihood_Function(double H_0, double omega_m0, double omega_q0,
   double alpha, double alpha_x, double m, double data[],
    double error[], int dataSize){
    /*
    This function returns the likelihood function given the set of
    cosmological parameters and the experimental data.

    Input
    - H_0 (double): Hubble parameter
    - (omega_m0, omega_q0, alpha, alpha_x, m) (floats): cosmological parameters
    - data (array): set of data from the observations
    - error (array): set of time_delay uncertainties

    Output:
    - L (double): likelihood function
    */
    double z_ds [dataSize]; // [x[0] for x in data]
    double z_ss [dataSize]; //  [x[1] for x in data]
    double delta_ts [dataSize]; // [x[2] for x in data]
    double theta_is [dataSize]; // [x[3] for x in data]
    double theta_js [dataSize]; // [x[4] for x in data]

    double acumulator = 0;
    for (int ii=0; ii<dataSize; ii++){
      z_ds[ii] = data[ii*5];
      z_ss[ii] = data[1+ii*5];
      delta_ts[ii] = data[2+ii*5];
      theta_is[ii] = data[3+ii*5];
      theta_js[ii] = data[4+ii*5];
      double delay =time_delay(theta_is[ii], theta_js[ii], H_0,
        z_ds[ii], z_ss[ii], omega_m0, omega_q0, alpha, alpha_x, m);
      acumulator -= pow(delta_ts[ii]- delay,2)/ pow(error[ii],2);
    }
    return acumulator;
}

bool model_Prior(double H_0, double omega_m0, double omega_q0, double alpha,
  double alpha_x, double m){
    /*
    This prior distribution discards all the values that doesn't have
    physical meaning.

    Input
    - H_0 (double): Hubble parameter
    - (omega_m0, omega_q0, alpha, alpha_x, m) (doubles): cosmological parameters
    - means (array): ordered list with the means of all the parameters

    Output:
    - Prior value (bool): accept or reject
    */
    // Accept only valid cosmological parameters
    bool cond = (H_0<0) || (omega_m0>1) || (omega_q0>1) || (omega_m0<0) ||
    (omega_q0<0) || (alpha<0) || (alpha_x<0) || (m<0) || ((omega_m0+omega_q0)>1);

    if (cond){
      return 0;
    }
    return 1;
}

double* transition_Model(double meansAndDeviations[]){
    /*
    This function chooses the new values to be compared with the last values
    using a normal like distribution for all the parameters.

    Input:
    - means (list): list with all the means of the cosmological parameters
    - deviations (list): list with all the deviations of the cosmological parameters

    Output:
    - l (list): list with the new means and deviations for the cosmological parameters,
    structured as follows: new_means+new_deviations
    */
    double* l = new double[12];
    //- (H_0, omega_m0, omega_q0, alpha, alpha_x, m) (doubles): cosmological parameters
    for (int ii=0; ii<6; ii++){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      default_random_engine generator (seed);
      normal_distribution<double> distribution(meansAndDeviations[ii],
      meansAndDeviations[6+ii]);

      l[ii] = distribution(generator);
      l[ii+6] = meansAndDeviations[6+ii];

    }
    //l[1]=.3;
    //l[2]=0.7;
    //l[3]=1;
    //l[4]=0;
    //l[5]=0;
    return l;
}

bool acceptance_rule(double old, double renew){
    /*
    This function determines if accept or reject the new data, based on the
    detailed balance method.

    Input:
    - old (double): old value of detailed balance condition
    - new (double): new value of detailed balance condition

    Output:
    - (Bool): True if accepting, False if rejecting
    */
    if (renew>old){
      return 1;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    uniform_real_distribution<double> distribution(0.0,1.0);
    double accept = distribution(generator);
    return (accept < exp(renew-old));
}
//////// Working Zone ////////
void metropolis_Hastings(double param_init[],double data[], int dataSize,
   double error[], int accept_limit=-1, int decreaseDeviation=-1, int iterations =100)
   {
    /*
    This function implements the Metropolis Hastings method for obtaining
    the best possible parameters from a set of data.

    Input:
    - param_init (list): initial parameters (means and deviations)
    - data (array): experimental data
    - error (array): uncertainties of delay time
    - accept_limit (int>0, optional): stop when reaching len(accepted_values)==accept_limit
    - decreaseDeviation (int>0, optional) : decrease Deviations when not accepting
    after decreaseDeviation iterations
    - iterations (int>0, optional): number of iterations

    Output:
    - VOID
    */
    //Order of Data
    //H_0,omega_m0, omega_q0, alpha, alpha_x, m

    double* x = param_init;
    int trueTime = 0;
    int timesWithoutAccept=0;
    int accepted = 0;

    ofstream fileAccepted;
    ofstream fileRejected;
    ofstream fileLikeli;

    fileAccepted.open("tmpRes4/accepted.txt");
    fileRejected.open("tmpRes4/rejected.txt");
    fileLikeli.open("tmpRes4/likelihoodAccepted.txt");

    if (accept_limit==-1){
      for(int ii=0; ii<iterations; ii++){
            double* x_new =  transition_Model(x);
            if (model_Prior(x_new[0], x_new[1], x_new[2], x_new[3], x_new[4],
            x_new[5]) == 1){
              double x_lik = likelihood_Function(x[0], x[1], x[2], x[3],
                x[4], x[5], data, error, dataSize);

              double x_new_lik = likelihood_Function(x_new[0], x_new[1], x_new[2], x_new[3],
               x_new[4], x_new[5],data, error, dataSize);

              if (acceptance_rule(x_lik ,x_new_lik)){
                  x = x_new;
                  for (int jj=0; jj<12; jj++){
                    fileAccepted << x_new[jj] <<" ";
                  }
                  fileAccepted << "\n";
                  fileLikeli << x_new_lik << "\n";
                  timesWithoutAccept = 0;
                  trueTime = 0;
                  accepted += 1;
              }
              else{
                  for (int jj=0; jj<12; jj++){
                    fileRejected << x_new[jj] <<" ";
                  }
                  fileRejected << "\n";
                  timesWithoutAccept += 1;
                  trueTime +=1;
              }
            }else{
              timesWithoutAccept += 1;
              trueTime +=1;
            }

            if (decreaseDeviation !=-1){
                if (timesWithoutAccept > decreaseDeviation){
                  for(int jj=6; jj<12; jj++){
                    x[jj] = x[jj]/1.2;
                  }
                    timesWithoutAccept = 0;
                }
            }
            printf("Iteration %d/%d. Accepted Values: %d. Iterations without accepting: %d      \r",
             ii+1,iterations,accepted, trueTime);

          }
    }
    else{
        int ii=0;
        while(accepted<accept_limit){
          double* x_new =  transition_Model(x);
          if (model_Prior(x_new[0], x_new[1], x_new[2], x_new[3], x_new[4],
          x_new[5]) == 1){
            double x_lik = likelihood_Function(x[0], x[1], x[2], x[3],
              x[4], x[5], data, error, dataSize);


            double x_new_lik = likelihood_Function(x_new[0], x_new[1], x_new[2], x_new[3],
             x_new[4], x_new[5],data, error, dataSize);

            if (acceptance_rule(x_lik ,x_new_lik)){
                x = x_new;
                for (int jj=0; jj<12; jj++){
                  fileAccepted << x_new[jj] <<" ";
                }
                fileAccepted << "\n";
                fileLikeli << x_new_lik << "\n";
                timesWithoutAccept = 0;
                trueTime = 0;
                accepted += 1;
            }
            else{
                for (int jj=0; jj<12; jj++){
                  fileRejected << x_new[jj] <<" ";
                }
                fileRejected << "\n";
                timesWithoutAccept += 1;
                trueTime +=1;
            }
          }else{
            timesWithoutAccept += 1;
            trueTime +=1;
          }

          if (decreaseDeviation !=-1){
              if (timesWithoutAccept > decreaseDeviation){
                for(int jj=6; jj<12; jj++){
                  x[jj] = x[jj]/1.2;
                }
                  timesWithoutAccept = 0;
              }
          }
          printf("Iteration %d. Accepted Values: %d/%d. Iterations without accepting: %d      \r",
           ii+1,accepted, accept_limit, trueTime);
           ii += 1;
         }
    }
    fileAccepted.close();
    fileRejected.close();
    fileLikeli.close();
}
