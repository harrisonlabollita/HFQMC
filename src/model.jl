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
