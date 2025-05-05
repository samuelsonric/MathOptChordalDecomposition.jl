module MathOptChordalDecomposition

using Base: oneto
using CliqueTrees
using CliqueTrees: EliminationAlgorithm
using LinearAlgebra
using SparseArrays

import MathOptInterface as MOI

const DICT = Dict{
    MOI.ConstraintIndex{
        MOI.VectorAffineFunction{Float64},
        MOI.PositiveSemidefiniteConeTriangle},
    Tuple{
        MOI.ConstraintIndex{
            MOI.VectorAffineFunction{Float64},
            MOI.Zeros},
        Vector{
            MOI.ConstraintIndex{
                MOI.VectorOfVariables,
                MOI.PositiveSemidefiniteConeTriangle,
            },
        },
        Vector{Vector{Int}},
        Int,
    },
}

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    outer_to_inner::DICT

    function Optimizer(optimizer_factory)
        return new(
            MOI.instantiate(
                optimizer_factory;
                with_cache_type = Float64,
                with_bridge_type = Float64,
            ),
            DICT(),
        )
    end
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.inner)
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

MOI.optimize!(model::Optimizer) = MOI.optimize!(model.inner)

const _ATTRIBUTES = Union{
    MOI.AbstractConstraintAttribute,
    MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute,
    MOI.AbstractVariableAttribute,
}

function MOI.set(model::Optimizer, attr::_ATTRIBUTES, args...)
    MOI.set(model.inner, attr, args...)
    return
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, args...)
    if MOI.is_set_by_optimize(attr)
        msg = "MOCD does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector{T}) where {T}
    if MOI.is_set_by_optimize(attr)
        msg = "MOCD does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    return MOI.get.(model, attr, arg)
end

### AbstractOptimizerAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractOptimizerAttribute)
    return MOI.supports(model.inner, arg)
end

function MOI.set(model::Optimizer, attr::MOI.AbstractOptimizerAttribute, value)
    MOI.set(model.inner, attr, value)
    return
end

function MOI.get(model::Optimizer, attr::MOI.AbstractOptimizerAttribute)
    return MOI.get(model.inner, attr)
end

### AbstractModelAttribute

function MOI.supports(model::Optimizer, arg::MOI.AbstractModelAttribute)
    return MOI.supports(model.inner, arg)
end

### AbstractVariableAttribute

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.inner, x)
end

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.inner)

function MOI.delete(model::Optimizer, x::MOI.VariableIndex)
    MOI.delete(model.inner, x)
    return
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(model.inner, arg, MOI.VariableIndex)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    indices::Vector{<:MOI.VariableIndex},
    args::Vector{T},
) where {T}
    MOI.set.(model, attr, indices, args)
    return
end

### AbstractConstraintAttribute

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(model.inner, ci)
end

function MOI.supports(
    model::Optimizer,
    arg::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.supports(model.inner, arg, MOI.ConstraintIndex{F,S})
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    indices::Vector{<:MOI.ConstraintIndex},
    args::Vector{T},
) where {T}
    MOI.set.(model, attr, indices, args)
    return
end

function MOI.get(model::Optimizer, ::Type{MOI.VariableIndex}, args...)
    return MOI.get(model.inner, MOI.VariableIndex, args...)
end

function MOI.get(model::Optimizer, T::Type{<:MOI.ConstraintIndex}, args...)
    return MOI.get(model.inner, T, args...)
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex)
    MOI.delete(model.inner, ci)
    return
end

function MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
)
    return MOI.supports_constraint(model.inner, F, S)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.AbstractFunction,
    s::MOI.AbstractSet,
)
    return MOI.add_constraint(model.inner, f, s)
end

# Decomposition

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.VectorAffineFunction{T},
    s::MOI.PositiveSemidefiniteConeTriangle,
) where {T}
    # construct sparse matrices
    V, A, b = decode(f, s)
    
    # compute aggregate sparsity pattern
    pattern = sum(sparsitypattern, A; init=sparsitypattern(b))
    
    # compute tree decomposition
    label, tree = cliquetree(pattern; alg=MF())
    
    # compute cliques
    cliques = map(tree) do clique
        return sort!(label[clique])
    end
    
    # terms
    indices = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}[]
    terms = MOI.VectorAffineTerm{T}[]
    
    for clique in cliques
        m = length(clique)
        U = MOI.add_variables(model, m * (m + 1) ÷ 2)
        
        for j in oneto(m), i in oneto(j)
            v = U[idx(i, j)]
            push!(terms, MOI.VectorAffineTerm(idx(clique[i], clique[j]), MOI.ScalarAffineTerm(-1.0, v)))
        end
        
        push!(indices, MOI.add_constraint(model.inner, MOI.VectorOfVariables(U), MOI.PositiveSemidefiniteConeTriangle(m)))
    end
    
    for (v, a) in zip(V, A), j in axes(a, 2)
        for p in nzrange(a, j)
            i = rowvals(a)[p]
            i > j && break
            x = nonzeros(a)[p]
            push!(terms, MOI.VectorAffineTerm(idx(i, j), MOI.ScalarAffineTerm(x, v)))
        end
    end
    
    # constants
    n = size(b, 2)
    constants = zeros(T, n * (n + 1) ÷ 2)
    
    for j in axes(b, 2)
        for p in nzrange(b, j)
            i = rowvals(b)[p]
            i > j && break
            x = nonzeros(b)[p]
            constants[idx(i, j)] = x
        end
    end
    
    index = MOI.add_constraint(model.inner, MOI.VectorAffineFunction(terms, constants), MOI.Zeros(n * (n + 1) ÷ 2))
    outer = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.PositiveSemidefiniteConeTriangle}(index.value)
    model.outer_to_inner[outer] = (index, indices, cliques, n)
    return outer
end

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{F,S},
) where {
    F<:MOI.VectorAffineFunction{Float64},
    S<:MOI.PositiveSemidefiniteConeTriangle,
}
    return length(model.outer_to_inner)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{F,S},
) where {
    F<:MOI.VectorAffineFunction{Float64},
    S<:MOI.PositiveSemidefiniteConeTriangle,
}
    return sort!(collect(keys(model.outer_to_inner)); by = c -> c.value)
end

function MOI.get(
    model::Optimizer,
    attribute::MOI.ConstraintPrimal,
    index::MOI.ConstraintIndex{F,S},
) where {
    F<:MOI.VectorAffineFunction{Float64},
    S<:MOI.PositiveSemidefiniteConeTriangle,
}
    index, indices, cliques, n = model.outer_to_inner[index]
    result = zeros(Float64, n * (n + 1) ÷ 2)

    for (index, clique) in zip(indices, cliques)
        m = length(clique)
        vector = MOI.get(model.inner, attribute, index)

        for j in oneto(m), i in oneto(j)
            result[idx(clique[i], clique[j])] += vector[idx(i, j)]
        end
    end

    return result
end

function MOI.get(
    model::Optimizer,
    attribute::MOI.ConstraintDual,
    index::MOI.ConstraintIndex{F,S},
) where {
    F<:MOI.VectorAffineFunction{Float64},
    S<:MOI.PositiveSemidefiniteConeTriangle,
}
    index, indices, cliques, n = model.outer_to_inner[index]
    result = zeros(Float64, n * (n + 1) ÷ 2)

    for (index, clique) in zip(indices, cliques)
        m = length(clique)
        vector = MOI.get(model.inner, attribute, index)

        for j in oneto(m), i in oneto(j)
            result[idx(clique[i], clique[j])] = vector[idx(i, j)]
        end
    end

    return result
end

# Utilities

# `(V, A, b) = decode(f, S)` satisfies
#    f(Vᵢ) = Aᵢ + b
# for all 1 ≤ i ≤ n.
function decode(f::MOI.VectorAffineFunction{T}, S::MOI.PositiveSemidefiniteConeTriangle) where {T}
    n = S.side_dimension
    index = Dict{MOI.VariableIndex, Int}()
    
    # V, A
    V = MOI.VariableIndex[]
    AI = Vector{Int}[]
    AJ = Vector{Int}[]
    AX = Vector{T}[]
    
    for term in f.terms
        i = term.output_index
        v = term.scalar_term.variable
        x = term.scalar_term.coefficient
        
        if !haskey(index, v)
            push!(V, v)
            push!(AI, Int[])
            push!(AJ, Int[])
            push!(AX, T[])
            index[v] = length(V)
        end
            
        j = index[v]
        push!(AI[j], row(i))
        push!(AJ[j], col(i))
        push!(AX[j], x)
    end
    
    A = map(zip(AI, AJ, AX)) do (I, J, X)
        return sparse(I, J, X, n, n)
    end
    
    # b
    I = Int[]
    J = Int[]
    X = T[]
    
    for (i, x) in enumerate(f.constants)
        if !iszero(x)
            push!(I, row(i))
            push!(J, col(i))
            push!(X, x)
        end
    end
    
    b = sparse(I, J, X, n, n)
    return V, A, b
end

# `S = sparsitypattern(A)` is a binary matrix with the same
# sparsity pattern as A.
function sparsitypattern(A::AbstractMatrix{T}) where {T}
    S = sparse(Symmetric(A, :U))
    nonzeros(S) .= one(T)
    return S
end

# upper triangular indices
# idx ∘ (row, col) = id
# (row, col) ∘ idx = id
function idx(i::Int, j::Int)
    @assert i <= j
    x = i + j * (j - 1) ÷ 2
    return x
end

function col(x::Int)
    j = ceil(Int, (sqrt(1 + 8x) - 1.0) / 2)
    return j
end

function row(x::Int)
    j = col(x)
    i = x - j * (j - 1) ÷ 2
    return i
end

end
