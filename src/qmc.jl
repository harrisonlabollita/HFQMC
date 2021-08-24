"""
function Δ(g::Matrix, m::Tuple, ising_config::Matrix)
    g            - green's function matrix
    iτ           - time slice 
    ising_config - configuration of ising fields
"""
function Δ(g::Matrix, iτ::Int64, ising_conifg::Matrix)
    a = [exp(-2*ising_config[iτ]) -1, exp(2*ising_config[iτ]) -1]
    Δup = 1+a[1]*(1-g[1][p,p])
    Δdn = 1+a[2]*(1-g[2][p,p])
    return (Δup*Δdn, a)
end


function cleanupdate!(g::Matrix, g0::Matrix, ising_config::Array)
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



function accept!(iτ::Int64, g::Matrix, a::Array, ising_config::Array, x0, x1)
    ising_config[iτ] *= -1 # flip the spin
    L = length(ising_config)
    for (i, s) in enumerate([1, -1])
        b = a[i]/(1+a[i]*(1-g[i][iτ,iτ]))
        x0 = copy(g[i][:, iτ])
        x0[iτ] -= 1
        x1 = g[i][iτ,:]
        # perform rank1 update
        BLAS.ger!(b, x0, x1, g[i])
    end
end


function save(g::Matrix, nn, nd, Ḡτ)
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

function run!(ω, G0iω, τ, config, λ)
    L = length(config)
    vn = λ .* config
    G0τ = invFourier(G0iω, ω, τ)
    g0 = g0_2D(τ, G0τ)
    g = [zeros(L, L), zeros(L, L)]
    cleanupdate!(g, g0, vn)
    x0 = zeros(L)
    x1 = zeros(L)
    accepted = 0
    stored = 0
    nn = 0
    nd = 0
    Ḡτ = zeros(L+1)
    for istep=1:nsteps
        iτ = 1 + convert(Int, round(rand()*(L-1), digits=0))
        ρr, a = Δ(iτ, g, vn)
        if ρr < 0
            @warn "!!! Sign problem !!!"
        end
        if abs(ρr) > rand() # metroplis
            accept!(iτ, g, a, vn, x0, x1)
            accepted += 1
        end
        
        if istep>nwarmup && (istep-nwarmup)%measure == 0
            nn, nd, Ḡτ = save(g, nn, nd, Ḡτ)
            stored +=1
        end
        
    end
    Ḡτ ./= stored
    nn /= stored
    nd /= stored
    println("density = $(nd), double-occ = $(nn)")
    Giω = Fourier(Ḡτ, τ, β, ω)
    return G0iω, Giω, Ḡτ, config, G0τ
end
