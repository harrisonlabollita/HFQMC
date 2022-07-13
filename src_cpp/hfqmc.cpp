// HF QMC

#include <iostream>
#include <vector>
#include <complex>
#include <stdlib.h>
#include <time.h>  
#include<algorithm>
#include<iterator>

#include "greens.hpp"

using namespace std;



// QMC class for Hirsch-Fye QMC
class QMC {
	private:
		float beta;
		int N;
		int L;
		float U;
		long int nsteps;
		int nwarmup;
		int measure;
		float lambda;
		vector<int> ising_spins;

		void init_ising_spins () {
			for (int i=0; i<L; i++) {
				ising_spins.push_back((double) rand()/RAND_MAX > 0.5 ? 1 : -1);
			}
		};

		vector<vector<float>> g0_2d() {
			vector<vector<float>> g0(L, vector<float>(L, 0.0));
				for (int i=0; i<L; i++) {
					for (int j=0; j<L; j++) {
						g0[i][j] = (i >= j) ? -G0_tau.data[i-j] : G0_tau.data[L+i-j];
					}
				}
				return g0;
		};

		void cleanupdate(vector<float> &g) {

			vector<vector<float>> A(L, vector<float>(L, 0.0));
			vector<float> spins = {1, -1};
			for (int s=0; s<spins.size(); s++){
				vector<float> a(L, 0.0);
				for (int ia=0; ia < L; ia++){
					a[ia] = exp(lambda*ising_spins[ia]*spins[s]) - 1.0;
				}
				for (int l1=0; l1<L; l1++) {
					for (int l2=0; l2<L; l2++){
						A[l1][l2] = -1*G0_tau_2d[l1][l2]*a[l2];
					}
					A[l1][l1] += 1+a[l1];
				}
				// call out to blas/lapack here
				g[ia]; // TODO: write this part
			}
		}

	public:

		GFiw G0_iw; // G0(iw)
		GFtau G0_tau; // G0(tau)

		vector<vector<float>> G0_tau_2d; // G0 (tau ) L x L matrix for QMC

		QMC (float beta=16.0, int N=1500, int L=64, float U=2.0, long int nsteps=100000) {
				beta = beta;
				N    = N;
				L    = L;
				U    = U;
				nsteps = nsteps;
				nwarmup = 10*L;
				measure = 100*L;
		};
		void init_G0_iw() {
			Bethe G0 (beta, N); // Hard coded to be the Bethe lattice G0;
			G0_iw = G0;
		};
		void init_G0_tau() {
			G0_tau = G0_iw.inverse_fourier(L);
		};
		void init_G0_tau_2d() {
			G0_tau_2d = g0_2d();
		};

		float det_ratio(int p, vector<vector<vector<float>>> g)
		{
			ising_spins[p] *= -1;
			float vn = lambda*ising_spins[p];
			vector<float> a(2, 0.0);
			a[0] = exp(-2*vn)-1.0f;
			a[1] = exp(2*vn)-1.0f;
			float det_up=1.0f+a[0]*(1-g[0][p][p]);
			float det_dn=1.0f+a[1]*(1-g[1][p][p]);
			return det_up*det_dn;
		}

		void accept_move(int p, vector<float> a)
		{
			//accept the move and update the Green's function
			vn[p] *= 1;
			vector<int> spins = {1, -1}
			for(int s=0; s<spins.size(); s++){
				float b = a[s]/(1 + a[s]*(1-g[s][p,p])); // <- g is a vector of two matrices? should we just split into gup and gdn?
				vector<float> x0(g[s].size(), 0.0);
				for (int k=0; k<x0.size(); k++) x0[k] = g[s][k,p];
				x0[p] -= 1.0;
				vector<float> x1(g[s].size(), 0.0);
				for (int k=0; k<x0.size(); k++) x1[k] = g[s][p,k];
				// call to blas here dger

			}
		}
		void save_measure(void) {
		//TODO: save measurement
		
		}
		void run_hfqmc(void) {
		//TODO: run the hfqmc
		
		}


};


int main() {
	QMC qmc;
	//qmc.init_G0_iw();
	return 0;
}



/*

int main() {
	srand(time(0));
	///////////////////////////////////////////////////////
	const float beta  = 16.0; // inverse temperature
	const int    N    = 1500; // number of matsubara points
	const int    L    = 64;   // number of time slices
	const float  U    = 2.0;  // interaction U
	///////////////////////////////////////////////////////
	const int nsteps    = 100000; // number of MC steps
	const int std_out   = 10000; // how often to print info
	
	int nwarm0  = 100; // number of warmup sweeps
	int nmeas0 = 10; // how often to measure
	
	int nwarmup =  int(nwarm0*L);
	int nmeasure = int(nmeas0*L);

	cout << "        HF QMC             " << endl;
	cout << "===========================" << endl;
	cout << "beta    = " << beta << endl;
	cout << "U       = " << U << endl;
	cout << "dtau    = " << float(L)/float(N) << endl;
	cout << "===========================" << endl;

	noninteractingG0 g0start (beta, N);

	vector<int> S_ising = isingFields(L);
	
	vector<complex<float>> G0iom;
	copy(g0start.Gf.begin(), g0start.Gf.end(), back_inserter(G0iom));
	vector<float> tau = linspace(0.0, beta, L+1);

	vector<float> G0tau = InverseFourier(G0iom, g0start.mesh, tau);

	return 0;
}


*/
