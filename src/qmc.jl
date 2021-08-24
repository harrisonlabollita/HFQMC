function cleanupdate!(g, g0, vn)
    L = size(g[1])[1]
    A = zeros(L, L)
    for (i, spin) in enumerate([1, -1])
        a = exp.(vn .* spin) .-1 
        for l1=1:L
            for l2=1:L
                A[l1, l2] = -g0[l1,l2]*a[l2]
            end
            A[l1, l1] += 1 .+ a[l1]
        end
        g[i] = A\g0
    end
end

function Δ(p, g, vn)
    a = [exp(-2*vn[p])-1, exp(2*vn[p])-1]
    Δup = 1+a[1]*(1-g[1][p,p])
    Δdn = 1+a[2]*(1-g[2][p,p])
    return (Δup*Δdn, a)
end

function accept!(p, g, a, vn, x0, x1)
    vn[p] *= -1 # flip the spin
    L = length(vn)
    for (i, s) in enumerate([1, -1])
        b = a[i]/(1+a[i]*(1-g[i][p,p]))
        x0 = copy(g[i][:,p])
        x0[p] -= 1
        x1 = g[i][p,:]
        BLAS.ger!(b, x0, x1, g[i])
    end
end

function save(g, nn, nd, Ḡτ)
    L = size(g[1])[1]
    Gτ = zeros(L+1)
    for s=1:2
        for i=1:L
            for j=1:L
                if (i>=j)
                    Gτ[i-j+1] -= g[s][i,j]
                else
                    Gτ[L+i-j+1] += g[s][i,j]
                end
            end
        end
    end
    Gτ .*= 1/(2*L)
    Gτ[L+1] = -Gτ[1] -1
    nnd = 0 # density
    for l=1:L
        nnd += (1-g[1][l,l] + (1-g[2][l,l]))
    end
    nnd /= L
    nnt = 0 # double-occ
    for l=1:L
        nnt += (1-g[1][l,l])*(1-g[2][l,l])
    end
    nnt /= L
    return nn.+nnt, nd.+nnd, Ḡτ .+ Gτ
end

function run(ω, G0iω, τ, config, λ)
    L = length(config)
    vn = λ .* config
    G0τ = invFourier(G0iω, ω, τ)
    g0 = g0_2D(τ, G0τ)
    
    #       ↑             ↓
    g = [zeros(L, L), zeros(L, L)]
    cleanupdate!(g, g0, vn)
    x0 = zeros(L)
    x1 = zeros(L)
    accepted = 0
    stored = 0
    st1 = 0
    st2 = 0
    st3 = 0
    
    nn = 0
    nd = 0
    
    Ḡτ = zeros(L+1)
    @printf "%2s %6s %6s %8s %8s %8s\n" "no." "accpt" "strd" "t-try" "t-accpt" "t-measure"
    for istep=1:nsteps
        t1 = time_ns()/1e9
        
        p = 1 + convert(Int, round(rand()*(L-1), digits=0))
        ρr, a = Δ(p, g, vn)
        if ρr < 0
            @warn "!!! Sign problem !!!"
        end
        t2 = time_ns()/1e9
        
        st1 += t2-t1
        if abs(ρr) > rand() # metroplis
            accept!(p, g, a, vn, x0, x1)
            accepted += 1
        end
        t3 = time_ns()/1e9
        st2 += t3-t2
        
        if istep>nwarmup && (istep-nwarmup)%measure == 0
            nn, nd, Ḡτ = save(g, nn, nd, Ḡτ)
            stored +=1
        end
        t4 = time_ns()/1e9
        st3 += t4-t3
        
        if istep % ncout == 1
	    @printf "%2i %6d %6d %8.4f %8.4f %8.4f\n" istep/ncout accepted stored st1/istep*1e5 st2/istep*1e5 st3/istep*1e5
        end
    end
    Ḡτ ./= stored
    nn /= stored
    nd /= stored
    println("density = $(nd), double-occ = $(nn)")
    Giω = Fourier(Ḡτ, τ, β, ω)
    return G0iω, Giω, Ḡτ, config, G0τ
end
