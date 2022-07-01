
#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <stdlib.h>
#include <time.h>  
#include <algorithm>
#include <iterator>

#include "utils.hpp"
using namespace std;

vector<float> linspace(float, float, int);

vector<float> linspace(float start, float stop, int size){
	vector<float> array;
	float delta  = (stop-start) / (size - 1);
	for (int i=0; i < size-1; ++i){
		array.push_back(start + delta*i);
	}
	array.push_back(stop);
	return array;
}

class GFtau {
	public:
		string        mesh_type;
		int           L;
		vector<float> mesh;
		vector<float> data;
		float beta;

		GFtau (float beta=16.0, int L = 64) {
			mesh_type = "im_time";
			mesh = linspace(0.0, beta, L+1);
			for (auto i=0; i<mesh.size(); i++){
				data.push_back(0.0);
			}
		}
		GFiw fourier(vector<float> iw, int m = 400) {
			// perform Fourier transform G(tau) -> G(iw)
			GFiw Giw(beta, m); // a fresh GF object
			for (int w=0; w<iw.size(); w++){
				double sum_re, sum_im = 0;
				for (int t=0; mesh.size()-1; t++) {
					double c0 = cos(mesh[t]*iw[w]), c1 = cos(mesh[t+1]*iw[w]);
					double s0 = sin(mesh[t]*iw[w]), s1 = sin(mesh[t+1]*iw[w]);
					double G0 = data[t], G1 = data[t+1];
					double dG = (G1-G0)/(mesh[t+1]-mesh[t]);
					sumim += (c0*G0-c1*G1 + dG*(s1-s0)/iw[w])/iw[w];
					sumre += (s1*G1-s0*G0 + dG*(c1-c0)/iw[w])/iw[w];
				}
				Giw.data[w] = complex<float> (sumre, sumim);
			}
			return Giw;
		}
};

// Green's function  on imaginary axis
class GFiw {
	public:
		string        mesh_type;
		int           N;
		float         beta;
		vector<float> mesh;
		vector<complex<float>> data;
		GFiw (float beta=16.0, int N = 1500) {
			mesh_type = "im_freq";
			beta = beta;
			N    = N;
		// initialize mesh and empty data structure	
		for (auto i=0; i <2*N; i++) {
			float w =  (2*i + 1.0)*pi/beta;
			mesh.push_back(w);
			data.push_back( complex<float> (0.0, 0.0));
			}
		}

		GFtau inverse_fourier(int L = 64) {
			GFtau Gtau(beta, L);
			for (int it=0; it<Gtau.mesh.size(); it++){
				float dsum = 0;
				for (int iw=0; iw<mesh.size(); iw++){
					dsum += cos(mesh[iw]*Gtau.mesh[it])*data[iw].real();
					dsum += sin(mesh[iw]*Gtau.mesh[it])*(data[iw].imag() + 1.0/mesh[iw]);
				}
				Gtau.data[it] += 2*dsum/beta - 0.5;
			}
			return Gtau;
		}
};


class Bethe : public GFiw {
	public:
		Bethe (float beta = 16.0, int N = 1500) {
			for (int i=0; i<2*N; i++){
				float w = mesh[i];
				data[i] = complex<float> (0.0, 2.0*(w - sqrt(w*w + 1.0)));
			}
		}
};




