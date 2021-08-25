abstract type GreensFunction end
abstract type GFImTime <: GreensFunction end
abstract type GFImFreq <: GreensFunction end

struct GreenImTime  <: GFImTime 
    τ::Array
    data::Array
end

struct GreenImFreq <: GFImFreq
    ω::Array
    β::Float64
    data::Array
end


function noninteractingG0(mesh::Int64=2050, β::Float64=16.)
    ωₙ = ((2 .* range(1, N, step=1) .+ 1) .* π ) ./β  # ωₙ = (2n +1)β
    # non-interacting Green's function for the Bethe lattice
    GreenImFreq(ωₙ, β, 2im .* (ωₙ - sqrt.(ω .* ω .+ 1)))
    # TODO- provide more cases or read from a file
    # This is pretty standard for a qmc solver
end

"""
Convert from G(τ₂-τ₁) = g[τ,τ'].
"""
function g0(G0::GreenImTime)
    L = length(G0.τ) - 1
    g0  = zeros(L, L)
    for i=1:L
        for j=1:L
            g0[i,j] = i>=j ? -G0.data[(i-j)+1] : G0.data[L+(i-j)]
        end
    end
    g0
end

"""
Compute the inverse fourier transform of G(iω).
"""
function invFourier(τ, G::GFImFreq)
    β = G.beta
    Gτ = zeros(length(τ))
    for (it, t) in enumerate(τ)
        sum = 0
        for (iωₙ, val) in enumerate(G.mesh)
            sum  += cos(val*t)*real(G.data[iωₙ]) + sin(val*t)*(imag(G.data)[iωₙ] + 1/val)
        end
        Gτ[it] = 2*sum/β - 0.5
    end
    GFImTime(τ, Gτ)
end
            
"""
Compute the fourier transfrom of G(τ)
"""
function Fourier(G::GFImTime, ω::Array, β::Float64, mm::Int64=400)
    L = length(G.τ)-1
    # use interpolation to reduce the noise
    gtk = CubicSplineInterpolation(G.τ[1]:(G.τ[2]-G.τ[1]):G.τ[end],G.data)
    dim = convert(Int64, mm*L+1)
    tt = collect(range(0, β, length=dim))
    gtt = gtk(tt)
    Giω = Array{Complex}(undef, length(ω))
    for (iω, oω) in enumerate(ω)
        rp = gtt .* cos.(tt.*oω)
        ip = gtt .* sin.(tt.*oω)
        Giω[iω] = simpson(tt, rp) + 1im*simpson(tt, ip)
    end
    GreenImFreq(ω, β, Giω)
end
