

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

