abstract type GreensFunction end
abstract type GImTime <: GreensFunction end
abstract type GImFreq <: GreensFunction end

struct GImTime  <: GImTime 
    τ::Array
    data::Array
end

struct GImFreq <: GImFreq
    ω::Array
    β::Float64
    data::Array
end


function noninteractingG0(mesh::Int64=2050, β::Float64=16.)
    ωₙ = ((2 .* range(1, N, step=1) .+ 1) .* π ) ./β  # ωₙ = (2n +1)β
    # non-interacting Green's function for the Bethe lattice
    GImFreq(ωₙ, β, 2im .* (ωₙ - sqrt.(ω .* ω .+ 1)))
end


function g0_2D(τ, G0)
    L = length(τ)-1
    g0 = zeros(L, L)
    for i=1:L, j=1:L
        if i>=j
            g0[i,j] = -G0[(i-j)+1]
        else
            g0[i,j] = G0[L+(i-j)]
        end
    end
    g0
end

function noninteracting_G0(β, N)
    ω = collect(range(1, 2*N, step=2)) .* π/β
   (ω, 2im*(ω-sqrt.(ω.*ω .+1)))
end


function invFourier(G::GreensFunction)
    β = G.beta
    Gτ = zeros(length(τ))
    for (it, t) in enumerate(τ)
        sum = 0
        for (iωₙ, val) in enumerate(G.mesh)
            sum  += cos(val*t)*real(G.data[iωₙ]) + sin(val*t)*(imag(G.data)[iωₙ] + 1/val)
        end
        Gτ[it] = 2*sum/β - 0.5
    end
    GImTime(τ, Gτ)
end
            

function invFourier(Giω, ω, τ)
    β = π/ω[1]
    Gτ = zeros(length(τ))
    for (it, t) in enumerate(τ)
        dsum = 0
        for (iω, oω) in enumerate(ω)
            dsum += cos(oω*t)*real(Giω[iω]) + sin(oω*t)*(imag(Giω)[iω]+1/oω)
        end
        Gτ[it] = 2*dsum/β - 0.5
    end
    Gτ
end;

function Fourier(Gτ, τ, β, ω, mm=400)
    L = length(τ)-1
    gtk = CubicSplineInterpolation(τ[1]:(τ[2]-τ[1]):τ[end],Gτ)
    dim = convert(Int64, mm*L+1)
    tt = collect(range(0, β, length=dim))
    gtt = gtk(tt)
    Giω = Array{Complex}(undef, length(ω))
    for (iω, oω) in enumerate(ω)
        rp = gtt .* cos.(tt.*oω)
        ip = gtt .* sin.(tt.*oω)
        Giω[iω] = simpson(tt, rp) + 1im*simpson(tt, ip)
    end
    Giω
end

function diff_Gf(ω, Gold, Gnew, weight=0.5)
	return sum(abs.((1 ./ (abs.(ω).^(weight)))) .* norm(Gnew .- Gold))
end
