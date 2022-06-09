#include "neatpp.hh"
#include <chrono>
using namespace std;

int main(int argc, char *argv[]) {

    // Default number of threads
    size_t nthreads = 4;

    // Input parameters
    double G0 = 4.1883992866138;
    double G2 = 0.0;
    double I2 = 0.0;
    double nfp = 2;
    double iota = -0.42047335182825846;
    double iotaN = -0.42047335182825846;
    const vector<double> phi_grid = {0.0, 0.23444962, 0.34976872, 0.57349475, 0.78551814,
        0.88679204, 1.08037373, 1.26388276, 1.4405271, 1.52745846, 1.70106555, 1.87770989,
        1.96838578, 2.15658791, 2.35607451, 2.46051567, 2.6786455, 2.90714304, 3.02397574, 3.14159265};
    double B0 = 4;
    double B1c = 2.56;
    vector<double> B20 = {0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64,
        0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64, 0.64};
    double B2c = -0.00322;
    double beta1s = 0.0;
    double charge = 2;
    double mass = 4;
    double energy = 3520000.0;
    size_t nlambda_trapped = 14;
    size_t nlambda_passing = 3;
    double r0 = 0.03;
    double r_max = 0.08;
    size_t ntheta = 16;
    size_t nphi = 8;
    size_t nsamples = 800;
    double Tfinal = 0.00001;
    
    // Read number of threads from argv if it exists
    if(argc==2)
        nthreads = stoi(argv[1]);
    else if(argc>2)
        throw "There is only one optional parameter that is the number of threads.";

    // Run function and time it
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    gc_solver_qs_partial_ensemble(
        G0, G2, I2, nfp, iota, iotaN,
        phi_grid, B0, B1c, B20, B2c,
        beta1s, charge, mass, energy,
        nlambda_trapped, nlambda_passing,
        r0, r_max, ntheta, nphi, nsamples,
        Tfinal, nthreads);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "nthreads = " << nthreads << " -> " <<  (chrono::duration_cast<chrono::microseconds>(end - begin).count()) /1000000.0 << "s" << endl;

  return 0;
}