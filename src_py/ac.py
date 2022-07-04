#!/usr/bin/env python3

# play around with some analytic continuation code
# goal shape into class structure and interface witrh TRIQS
import numpy as np


def calc_K_matrix(M, N, ωmax, β):
    K = np.zeros((M,N),dtype=float)

    dτ = β/(M-1)
    dω = 2*ωmax/(N-1)
    for iω in range(N):
        ω = iω*dω-ωmax
        for iτ in range(M):
            τ = iτ*dτ
            if τ*ω > 50.0:
                et1=np.exp(50.0)
            else:
                et1 = np.exp(τ*ω)
            if (τ-β)*ω > 50.0:
                et2 = np.exp(50.0)
            else:
                et2 = np.exp((τ-β)*ω)
            K[iτ,iω] = 1.0/(et1+et2)
    return K

