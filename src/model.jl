abstract type HubbardModel end

struct model <: HubbardModel
    sites::Int64
    neighbors::Array
    t::Float64
end

"""
populate the Hubbard Model struct.
"""
function model(L::Int64=2, t::Float64=-1.0)
    sites::Float64 = L*L
    lattice = reshape(collect(1:sites), L, L)
    neighbors = vcat(circshift(lattice, (-1,0))[:]',
                     circshift(lattice, (0,-1))[:]',
                     circshift(lattice, (1,0))[:]',
                     circshift(lattice, (0,1))[:]')
    model(sites, neighbors, t)
end


"""
build the hopping matrix for a LxL square lattice
"""
function hopping_matrix(model::HubbardModel)
    K = zeros(Float64, model.sites, model.sites)
    for i=1:model.sites
        K[i, model.neighbors[:, i]] .+= model.t
    end
    K
end

""" construct the interaction matrix
Vσˡ(s)_il;jl'=λσsᵢₗδll'δij
"""
function interaction_matrix(model, ising_config, τslice)
    λ = arccosh(exp(0.5*model.Δτ*U))
    V = [zeros(Float64, model.sites, model.sites), 
         zeros(Float64, model.sites, model.sites)]
    for (s, σ) in enumerate([-1, 1])
        for i=1:model.sites
            for j=1:model.sites
            if i == j
                V[s][i,j] += λ*ising_config[i, τslice]
            end
        end
    end
    V
end

