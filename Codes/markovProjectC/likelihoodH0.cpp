#include "markovDelay.h"
#include <fstream>
using namespace std;

int main(void){
  double h = 0.01;
  int N = 40/h;
  double* H_0 = new double[N+1];

  double data[] = {0.68,0.96,907200.0,1.1635528346628865e-06,4.84813681109536e-07,
 0.77,2.32,22090320.0,1.015684661924478e-05, 5.328102355393801e-06,
  0.31, 1.72,
 1010880.0, 5.560812922326377e-06,4.6057299705405916e-06, 0.31,1.72,2160000.0,
 6.772847125100217e-06, 4.6057299705405916e-06, 0.42, 1.59, 4060800, 5.52687596464871e-06,
 1.21203420277384e-06,0.89, 2.51, 2246400, 3.248251663433891e-06, 1.5514037795505152e-06};

  double errorDt[] = {17280.0,
    3155760.0,
     103680.0, 138240.0, 518400, 432000}; //seconds

  ofstream file;
  file.open("tmpRes/likelihood.txt");
  for (int ii = 0; ii<= N; ii++){
    H_0[ii] = 40+h*ii;
    file << H_0[ii]<< " " <<likelihood_Function(H_0[ii], .3, .7, 1, 0, 0, data, errorDt, 5) << "\n";
  }
  file.close();

  return 0;
}
