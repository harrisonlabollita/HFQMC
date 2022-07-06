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
        if x > α:
            return x-α
        elif x < -α: 
            return x + α
        else:
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

    ξ1 = (1/λ)*(S*Gp) + μp*(zp-up) + μ*np.dot(V.T, (z-u))
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

def calculate(U, S, V, Gtau, μ, μp, λ, tol=1e-8, max_iter=10000):
    L = S.shape[0]
    N = V.shape[0]

    Gp = np.dot(U.T, Gtau)
    xp = np.zeros(L)
    zp = np.zeros(L)
    up = np.zeros(L)
    z = np.zeros(N)
    u = np.zeros(N)

    for it in range(max_iter):
        xp, zp, up, z, u = update(S, V, Gp, zp, up, z, u, μp, μ, λ)
        if it%1000 == 0 and it > 0:
            print("it = ", it,"\t", "ΔF = ", np.sum(np.abs(z-np.dot(V,xp))))
        if np.sum(np.abs(z - np.dot(V,xp))) < tol:
            return np.dot(V,xp)
    print("[WARNING] convergence never reached!")
    return np.dot(V, xp)

def setup_mock_Gtau():
    err = 1e-3
    Giw = GfImFreq(beta=100, indices=[0])
    Gw = GfReFreq(window=(-10,10), indices=[0])
    Giw << SemiCircular(1.0)-0.5*SemiCircular(0.5)
    Gw << SemiCircular(1.0)-0.5*SemiCircular(0.5)

    Gtau = GfImTime(beta=100, indices=[0], n_points=4001)
    Gtau.set_from_fourier(Giw)

    Gtau.data[:,0,0] += err * np.random.randn(len(Gtau.data))
    return Gtau, Gw


def estimate(Gtau, K, U, S, V):

    μ = 1.0
    μp = 1.0
    λest = 0.0
    Fχold = 0.0

    l0 = -6.0
    λ0 = 10**l0
    # need to check this ----------
    #                             |
    #                            \ /
    xout = calculate(U, S, V, Gtau.data[:,0,0].real, μ, μp, λ0)
    χ0 = np.dot(Gtau.data[:,0,0].real-np.dot(K,xout),Gtau.data[:,0,0].real-np.dot(K,xout))

    l1 = 1.0
    λ1 = 10**l1
    xout = calculate(U, S, V, Gtau.data[:,0,0].real, μ, μp, λ1)
    χ1 = np.dot(Gtau.data[:,0,0].real-np.dot(K,xout),Gtau.data[:,0,0].real-np.dot(K,xout))

    b = (np.log(χ0)-np.log(χ1))/(np.log(λ0)-np.log(λ1))
    a = np.exp(np.log(χ0)-b*np.log(λ0))

    lexps = np.linspace(l0,l1,25)
    for i, ll in enumerate(lexps):
        λ = 10**ll
        xout = calculate(U, S, V, Gtau.data[:,0,0].real, μ, μp, λ)
        χ = np.dot(Gtau.data[:,0,0].real-np.dot(K,xout),Gtau.data[:,0,0].real-np.dot(K,xout))
        Fχ = a*λ**b/χ
        print(i,"/",len(lexps),  "λest= ", λ, "error= ", χ)
        if i > 2:
            if Fχ > Fχold:
                λest = λ
                Fχold = Fχ
        else:
            Fχold = Fχ

    λ = λest
    print("appropriate λ : ", λ)
    print("calculating final G(ω)...")
    xout = calculate(U, S, V, Gtau.data[:,0,0].real, μ, μp, λ)
    Gout = np.dot(K,xout)
    return xout, Gout

if __name__ == "__main__":
    Gtau, Gw = setup_mock_Gtau()
    # plot A(ω) = -1/π Im G(ω)
    fig, ax = plt.subplots(1,2)
    ax[0].plot([t.real for t in Gtau.mesh], Gtau.data[:,0,0].real)
    ax[1].plot([w.real for w in Gw.mesh], (-1/np.pi)*Gw.data[:,0,0].imag)
    plt.show()
    #plt.plot((-1/np.pi)*Gw.data[:,0,0].imag)
    #plt.show()
    N = 1001
    beta = 100.0
    ωmin, ωmax = -10, 10
    K = calc_K_matrix(ωmin, ωmax, beta, shape=(Gtau.data.shape[0], N))
    M, N = K.shape
    start = time.time()
    U, S, V = calc_svd(K)
    stop = time.time()
    # drop small values
    L = len(np.where(S>1e-11)[0])
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
    #Aout = np.zeros(N)

    ρout, Gout = estimate(Gtau, K, U, S, V)

    #Aout = estimate(Gqmc, K, U, S, V)
    #Gout = np.dot(K, Aout)



