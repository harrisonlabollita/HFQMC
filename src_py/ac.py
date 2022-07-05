#!/usr/bin/env python3
import time
# play around with some analytic continuation code
# goal shape into class structure and interface with TRIQS?
import numpy as np
from triqs.gf import *
import matplotlib.pyplot as plt

# Ref for ADMM algorithm: 10.1103/PhysRevE.95.061302
# G = K A
# Gi ≡ G(τi) : vector
# Kij ≡ K+(τi, ωj)
# Aⱼ ≡ A(ωⱼ)*Δω
# The square error is 
# χ²(A) = 1/2 || G - KA ||²₂ (L₂ norm)
# find ρ such that χ²(A) < η
# 2 conditions: Aj >= 0, ∑ⱼ Aj = 1
# the constraints are applied to the diagonal components of the
# Green's function. The off-diagonal components, the non-negativity is not applied.

# Basis set 
# K = U S Vᵀ
# S = M x N diagonal matrix
# U and V are orthogonal matrices
# must drop vectors below a threshold for numerical stability

# define new vectors
# A' = Vᵀ A, G' = UᵀG
# χ²(A') = 1/2 || G' - SA' || = 1/2 ∑ₗ(G'l - sl A'l)²
# Cost function
# F(A') = 1/2 || G' - SA'|| + λ || A' || <- L1 norm

# ADMM algorithm
#  F̃(A', z', z) = 1/2λ || A' - S G' || - ν(⟨VA'⟩ - 1) + ||z'|| + lim γ->∞ ∑j Θ(-zj)
#  z' = A', z = V A'
#


def calc_K_matrix(ωmin, ωmax, β, shape, sign="fermion"):
    M, N = shape[0], shape[1]
    # K±(τ,ω) = exp(-τ ω) / ( 1 ± exp(-β ω) )
    # K+(τ,ω) = 1 / ( exp(τω) + exp((-β+τ)*ω) )
    # K-(τ,ω) = 1 / ( exp(τω) - exp((-β+τ)*ω) )
    # + = fermion
    # - = boson
    cutoff = 50.0
    K = np.zeros((M,N), dtype=float)

    ωmesh = np.linspace(ωmin, ωmax, N)
    τmesh = np.linspace(0, β, M)
    for iω,ω in enumerate(ωmesh):
        for iτ,τ in enumerate(τmesh):
            τω = τ*ω
            βmτω = (-β+τ)*ω
            expτω = np.exp(τω) if τω <  cutoff else np.exp(cutoff)
            expβmτω = np.exp(βmτω) if βmτω <  cutoff else np.exp(cutoff)
            if sign == "fermion":
                K[iτ,iω] = 1.0/(expτω+expβmτω)
            else:
                K[iτ,iω] = 1.0/(expτω-expβmτω)
    return K


def calc_svd(K):
    return np.linalg.svd(K)

def Pplus(z): 
    return np.array([max(zj,0) for zj in z], dtype=float)

def Salpha(x, α):
    def _Salpha(x):
        if x > α: return x-α
        if x < -α: return x + α
        return 0
    return np.array(list(map(_Salpha, x)))

def compute_nu(ξ1, ξ2, V):
    Vξ1 = np.sum(np.dot(V, ξ1))
    Vξ2 = np.sum(np.dot(V, ξ2))
    return (1 - Vξ1)/Vξ2

def update(S, V, Gp, zp, up, z, u, μp, μ, λ):
    L = S.shape[0]
    e = np.ones(V.shape[0]) # determine size 

    ξ1 = np.zeros(L,dtype=float);
    ξ2 = np.zeros(L,dtype=float);
   
    inv = np.zeros((L,L),dtype=float)
    for i in range(L): inv[i,i] = 1.0/(S[i]**2/λ + (μ+μp))

    ξ1 = (1/λ)*np.dot(S.T, Gp) + μp*(zp-up) + μ*np.dot(V.T, (z-u))
    ξ2 = np.dot(V.T, e)

    ξ1 = np.dot(inv, ξ1) 
    ξ2 = np.dot(inv, ξ2) 

    ν = compute_nu(ξ1, ξ2, V)
    xp = ξ1 + ν*ξ2

    # updates
    zp = Salpha(xp + up, 1/μp)
    up += (xp - zp)
    z = Pplus(np.dot(V, xp) + u)
    u += np.dot(V, xp) - z
    return xp, zp, up, z, u

def setup_mock_Gtau():
    err = 1e-5
    Giw = GfImFreq(beta=10, indices=[0])
    Gw = GfReFreq(window=(-10,10), indices=[0])
    Giw << SemiCircular(1.0)-0.5*SemiCircular(0.5)
    Gw << SemiCircular(1.0)-0.5*SemiCircular(0.5)

    Gtau = GfImTime(beta=10, indices=[0], n_points=2501)
    Gtau.set_from_fourier(Giw)

    Gtau.data[:,0,0] += err * np.random.randn(len(Gtau.data))
    return Gtau, Gw


if __name__ == "__main__":
    Gtau, Gw = setup_mock_Gtau()
    N = 1001
    beta = 10.0
    ωmin, ωmax = -10, 10
    K = calc_K_matrix(ωmin, ωmax, beta, shape=(Gtau.data.shape[0], N))
    M, N = K.shape
    start = time.time()
    U, S, V = calc_svd(K)
    stop = time.time()
    # drop small values
    L = len(np.where(S>1e-10)[0])
    print('-'*80)
    print("SVD computation time = ", (stop-start), "s")
    print('singular values: ')
    for i in range(L): print("S[", i, "] = ", S[i] )
    print('-'*80)

    # setup  arrays with new dimension L
    U = U[:,:L]
    S = S[:L]
    V = V[:,:L]

    Gout = np.zeros(M)
    Aout = np.zeros(N)

    Gp = np.dot(U.T, Gtau.data[:,0,0].real)

    xp = np.zeros(L)
    zp = np.zeros(L)
    up = np.zeros(L)
    z = np.zeros(N)
    u = np.zeros(N)

    μ = 1.0
    μp = 1.0
    λ = 10**-6.0

    update(S, V, Gp, zp, up, z, u, μp, μ, λ)

    #Aout = estimate(Gqmc, K, U, S, V)
    #Gout = np.dot(K, Aout)

    # plot A(ω) = -1/π Im G(ω)
    #plt.figure()
    #plt.plot((-1/np.pi)*Gw.data[:,0,0].imag)
    #plt.show()


