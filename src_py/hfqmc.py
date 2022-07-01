#!/usr/bin/env python3
import time
import numpy as np
from scipy import interpolate
from scipy import integrate
import scipy
from scipy.linalg import blas
import copy
import matplotlib.pyplot as plt


def noninteracting_G0(beta, N):
    mesh  = np.arange(1,2*N,2)*np.pi/beta
    return mesh, 2j*(mesh-np.sqrt(mesh**2 + 1))

def InverseFourier(Giom, mesh,  tau):
    beta = np.pi/mesh[0];
    Gtau = np.zeros(len(tau), dtype=float)
    for (it, t) in enumerate(tau):
        dsum = 0.0
        for (iw,w) in enumerate(mesh):
            dsum += np.cos(w*t)*Giom[iw].real + np.sin(w*t)*(Giom[iw].imag + 1./w)
        Gtau[it] = 2*dsum/beta - 0.5
    return Gtau

def Fourier(Gtau_avg, tau, beta, mesh, mm=500):
    gtk = interpolate.UnivariateSpline(tau, Gtau_avg,s=0)
    tt = np.linspace(0, beta, int(mm*L+1))
    gtt = gtk(tt)
    Giom = np.zeros(len(mesh), dtype=complex)
    for (iw,w) in enumerate(mesh):
        rp = gtt*np.cos(tt*w)
        ip = gtt*np.sin(tt*w)
        Giom[iw] = integrate.simps(rp,tt) + integrate.simps(ip,tt)*1j
    return Giom


def g0_2D(tau, G0):
    L = len(tau)-1
    g0 = np.zeros((L,L), dtype=float)
    for i in range(L):
        for j in range(L):
            if i>=j:
                g0[i,j] = -G0[i-j]
            else:
                g0[i,j] = G0[L+i-j]
    return g0


def clean_update(g, g0, vn):
    L = len(vn)
    A = np.zeros((L,L), dtype=float)
    for i,s in enumerate([1,-1]):
        a = np.exp(vn*s)-1.
        for l1 in range(L):
            for l2 in range(L): A[l1,l2] = -g0[l1,l2]*a[l2]
            A[l1,l1] += 1.0+a[l1]
        g[i] = scipy.linalg.solve(A,g0)


def det_ratio(p,g,vn):
    a = [np.exp(-2*vn[p])-1, np.exp(2*vn[p])-1]
    det_up = 1+a[0]*(1-g[0][p,p])
    det_dn = 1+a[1]*(1-g[1][p,p])
    return (det_up*det_dn, a)

def accept_move(p, g, a, vn, x0, x1):
    vn[p] *= -1
    L = len(vn)
    for i,s in enumerate([1,-1]):
        b = a[i]/(1+a[i]*(1-g[i][p,p]))
        x0 = copy.deepcopy(g[i][:,p])
        x0[p] -= 1.0
        x1 = g[i][p,:]
        blas.dger(b,x0,x1,a=g[i],overwrite_a=1)

def save_measure(g, nn, nd, Gtau_avg):
    L = len(g[0])
    Gt = np.zeros(L+1, dtype=float)
    for s in range(2):
        for i in range(L):
            for j in range(L):
                if (i>=j):
                    Gt[i-j] += -g[s][i,j]
                else:
                    Gt[L+i-j] += g[s][i,j]

    Gt *= 1/(2*L)
    Gt[L] = -Gt[0]-1.
    #density
    nnd = 0.0
    for l in range(L): nnd += (1-g[0][l,l]) + (1-g[1][l,l])
    nnd *= (1./L)
    #double occupancy
    nnt = 0.0
    for l in range(L): nnt += (1-g[0][l,l]) *(1-g[1][l,l])
    nnt *= (1./L)
    return (nn+nnt, nd+nnd, Gtau_avg+Gt)


def solve(mesh, G0iom, tau, s_ising, xlam):
    L = len(s_ising)
    vn = np.array(s_ising)*xlam
    
    G0tau = InverseFourier(G0iom, mesh, tau)

    g0 = g0_2D(tau, G0tau)

    g = [np.zeros((L,L),dtype=float, order='C'), np.zeros((L,L), dtype=float, order='C')]
    
    clean_update(g,g0,vn)

    x0,x1 = np.zeros(L,dtype=float),np.zeros(L,dtype=float)
    accepted, stored, st1, st2, st3=0,0,0,0,0

    nn, nd =0,0
    Gtau_avg = np.zeros(L+1, dtype=float)
    print("%-2s %-6s %-6s %8s %8s %8s" % ('#', 'accpt', 'strd', 't-try', 't-accpt', 't-measr'))
    for istep in range(nsteps):
        t1 = time.time()
        p = int(np.random.rand()*L)
        (rhor, a) = det_ratio(p, g, vn)
        if (rhor <0): print('sign problem !')
        t2 = time.time(); st1 += t2-t1
        if abs(rhor) > np.random.rand():  #metropolis
            accept_move(p, g, a, vn, x0, x1)
            accepted += 1
        t3 = time.time(); st2+= t3-t2
        if istep>nwarmup and (istep-nwarmup)%measure==0:
            (nn, nd, Gtau_avg) = save_measure(g, nn, nd, Gtau_avg)
            stored += 1

        t4 = time.time(); st3 += t4-t3

        if istep%ncout == 1:
            print("%-2d %-6d %-6d %8.4f %8.4f %8.4f" % (istep/ncout, accepted, stored, st1/istep*1e5, st2/istep*1e5, st3/istep*1e5))

    Gtau_avg *= 1./stored
    nn *= 1/stored;
    nd *= 1/stored;
    print('densty = ', nd, 'double-occupancy=', nn)

    Giom = Fourier(Gtau_avg, tau, beta, mesh, 400)
    return Giom, Gtau_avg, s_ising, G0tau


if __name__ == "__main__":
    beta = 32.
    N    = 1500
    L    = 64
    U    = 3.
    nwarm0 = 100
    nmeas0 = 10
    ncout  = 10000
    nsteps = 1000000
    nwarmup = int(nwarm0*L)
    measure = int(nmeas0*L)

    mesh, G0start = noninteracting_G0(beta,N)
    
    G0iom = G0start
    tau   = np.linspace(0, beta, L + 1)
    
    lam = np.arccosh(np.exp(0.5*(tau[1]-tau[0])*U)) # lambda

    np.random.seed(0)
    s_ising = np.array(np.sign(np.random.rand(L)-0.5), dtype=int)


    G_store=[]
    Giom_old = G0iom[:]

    for itt in range(10):
        Giom, Gtau_avg,s_ising,G0tau = solve(mesh, G0iom, tau, s_ising, lam)

        Sigma = 1/G0iom - 1/Giom

        G0iom_new = 1/(mesh*1j - 0.25*Giom)

        if itt>-1:
            G0iom = 0.5* G0iom_new[:] + (1-0.5)*G0iom[:]
        else:
            G0iom = G0iom_new[:]

        diff = sum(abs(Giom-Giom_old))
        Giom_old = Giom[:]
        print('itt=', itt, 'ΔG=', diff)
        G_store.append(Gtau_avg)

        fig, ax = plt.subplots(1,2)
        ax[0].plot(tau, Gtau_avg, 'o-', label='G(τ)')
        ax[0].plot(tau, G0tau, '-', label='G0(τ)')
        ax[0].legend()
        ax[1].plot(mesh, Sigma.imag, 'o-', mec='k',label='Σ(iω)')
        ax[1].set_xlim(0,2.5);ax[1].set_ylim(np.min(Sigma[:10].imag),0);
        ax[1].legend()
        plt.show()

    for i in range(len(G_store)):
        plt.plot(tau, G_store[i], 'o-', label='itt '+str(i))
    plt.legend(loc='best')
    plt.show()
