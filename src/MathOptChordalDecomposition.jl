module MathOptChordalDecomposition

using Base: oneto
using CliqueTrees
using CliqueTrees: EliminationAlgorithm
using LinearAlgebra
using Graphs
using SparseArrays

import MathOptInterface as MOI

struct Decomposition
    neqns::Int
    value::Vector{Int}
    label::Vector{Int}
    tree::CliqueTree{Int, Int}
end

"""
    Optimizer <: MOI.AbstractOptimizer

    Optimizer(inner::MOI.AbstractOptimizer; alg::CliqueTrees.EliminationAlgorithm=MF())

An optimizer that computes a chordal decomposition of each semidefinite constraint.
The elimination algorithm `alg` is used to compute the decomposition, and the optimizer
`inner` is used to solve the decomposed problem.

### Parameters
  - `inner`: inner optimizer
  - `alg`: elimination algorithm
"""
mutable struct Optimizer{A <: EliminationAlgorithm} <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    outer_to_inner::Dict{Int, Decomposition}
    alg::A

    function Optimizer(optimizer_factory, alg::A) where {A <: EliminationAlgorithm}
        return new{A}(
            MOI.instantiate(
                optimizer_factory;
                with_cache_type = Float64,
                with_bridge_type = Float64,
            ),
            Dict{Int, Decomposition}(),
            alg,
        )
    end
end

function Optimizer(optimizer_factory; alg::EliminationAlgorithm = MF())
    return Optimizer(optimizer_factory, alg)
end

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.inner)
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.inner)
end

function MOI.supports_incremental_interface(::Optimizer)
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(dest, src)
end

function MOI.optimize!(model::Optimizer)
    return MOI.optimize!(model.inner)
end

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
    #=
    if MOI.is_set_by_optimize(attr)
        msg = "MOCD does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    =#

    return MOI.get(model.inner, attr, args...)
end

function MOI.get(model::Optimizer, attr::_ATTRIBUTES, arg::Vector{T}) where {T}
    #=
    if MOI.is_set_by_optimize(attr)
        msg = "MOCD does not support querying this attribute."
        throw(MOI.GetAttributeNotAllowed(attr, msg))
    end
    =#

    return MOI.get.(model, attr, arg)
end

# -------------------------- #
# AbstractOptimizerAttribute #
# -------------------------- #

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

# ---------------------- #
# AbstractModelAttribute #
# ---------------------- #

function MOI.supports(model::Optimizer, arg::MOI.AbstractModelAttribute)
    return MOI.supports(model.inner, arg)
end

# ------------------------- #
# AbstractVariableAttribute #
# ------------------------- #

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    return MOI.is_valid(model.inner, x)
end

function MOI.add_variable(model::Optimizer)
    return MOI.add_variable(model.inner)
end

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

# --------------------------- #
# AbstractConstraintAttribute #
# --------------------------- #

function MOI.is_valid(model::Optimizer, ci::MOI.ConstraintIndex)
    return MOI.is_valid(model.inner, ci)
end

function MOI.supports(
        model::Optimizer,
        arg::MOI.AbstractConstraintAttribute,
        ::Type{MOI.ConstraintIndex{F, S}},
    ) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    return MOI.supports(model.inner, arg, MOI.ConstraintIndex{F, S})
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

#################
# Decomposition #
#################

function MOI.add_constraint(
        model::Optimizer,
        f::MOI.VectorAffineFunction{T},
        s::MOI.PositiveSemidefiniteConeTriangle,
    ) where {T}
    # construct sparse matrices
    V, A, b = decode(f, s); n = size(b, 2)

    # compute aggregate sparsity pattern
    pattern = sum(sparsitypattern, A; init = sparsitypattern(b))

    # compute tree decomposition
    label, tree = cliquetree(pattern; alg = model.alg)

    # terms
    value = Int[]
    terms = MOI.VectorAffineTerm{T}[]

    for bag in tree
        m = length(bag)
        U = MOI.add_variables(model, m * (m + 1) ÷ 2)

        for j in oneto(m), i in oneto(j)
            v = U[idx(i, j)]
            ii = label[bag[i]]
            jj = label[bag[j]]
            push!(terms, MOI.VectorAffineTerm(idx(ii, jj), MOI.ScalarAffineTerm(-1.0, v)))
        end

        index = MOI.add_constraint(model.inner, MOI.VectorOfVariables(U), MOI.PositiveSemidefiniteConeTriangle(m))
        push!(value, index.value)
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
    constants = zeros(T, n * (n + 1) ÷ 2)

    for j in axes(b, 2)
        for p in nzrange(b, j)
            i = rowvals(b)[p]
            i > j && break
            x = nonzeros(b)[p]
            constants[idx(i, j)] = x
        end
    end

    i = MOI.add_constraint(model.inner, MOI.VectorAffineFunction(terms, constants), MOI.Zeros(n * (n + 1) ÷ 2)).value
    model.outer_to_inner[i] = Decomposition(n, value, label, tree)
    return MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.PositiveSemidefiniteConeTriangle}(i)
end

function MOI.get(
        model::Optimizer,
        ::MOI.NumberOfConstraints{F, S},
    ) where {
        F <: MOI.VectorAffineFunction,
        S <: MOI.PositiveSemidefiniteConeTriangle,
    }
    return length(model.outer_to_inner)
end

function MOI.get(
        model::Optimizer,
        ::MOI.ListOfConstraintIndices{F, S},
    ) where {
        F <: MOI.VectorAffineFunction,
        S <: MOI.PositiveSemidefiniteConeTriangle,
    }

    indices = map(MOI.ConstraintIndex{F, S}, keys(model.outer_to_inner))
    sort!(indices; by = index -> index.value)
    return indices
end

function MOI.get(
        model::Optimizer,
        attribute::MOI.ConstraintPrimal,
        index::MOI.ConstraintIndex{F, S},
    ) where {
        T,
        F <: MOI.VectorAffineFunction{T},
        S <: MOI.PositiveSemidefiniteConeTriangle,
    }

    decomposition = model.outer_to_inner[index.value]
    neqns = decomposition.neqns
    value = decomposition.value
    label = decomposition.label
    tree = decomposition.tree
    result = zeros(T, neqns * (neqns + 1) ÷ 2)

    for (k, bag) in zip(value, tree)
        m = length(bag)

        part = MOI.get(
            model.inner,
            attribute,
            MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}(k),
        )

        for j in oneto(m), i in oneto(j)
            ii = label[bag[i]]
            jj = label[bag[j]]
            result[idx(ii, jj)] += part[idx(i, j)]
        end
    end

    return result
end

function MOI.get(
        model::Optimizer,
        attribute::MOI.ConstraintDual,
        index::MOI.ConstraintIndex{F, S},
    ) where {
        T,
        F <: MOI.VectorAffineFunction{T},
        S <: MOI.PositiveSemidefiniteConeTriangle,
    }

    decomposition = model.outer_to_inner[index.value]
    neqns = decomposition.neqns
    value = decomposition.value
    label = decomposition.label
    tree = decomposition.tree
    result = zeros(T, neqns * (neqns + 1) ÷ 2)
    W = zeros(T, neqns, neqns)

    for (k, bag) in zip(value, tree)
        m = length(bag)

        part = MOI.get(
            model.inner,
            attribute,
            MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.PositiveSemidefiniteConeTriangle}(k),
        )

        for j in oneto(m), i in oneto(m)
            ii = bag[i]
            jj = bag[j]
            W[ii, jj] = part[idx(i, j)]
        end
    end

    complete!(W, tree)

    for j in oneto(neqns), i in oneto(j)
        ii = label[i]
        jj = label[j]
        result[idx(ii, jj)] = W[i, j]
    end

    return result
end

# --------- #
# Utilities #
# --------- #

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

# Chordal Graphs and Semidefinite Optimization
# Vandenberghe and Andersen
# Algorithm 10.2: Positive semidefinite completion
function complete!(W::Matrix, tree::CliqueTree)
    n = nv(FilledGraph(tree))
    η = sizehint!(Int[], n)
    marker = zeros(Int, n)

    for bag in reverse(eachindex(tree))
        α = separator(tree, bag)
        ν = residual(tree, bag)
        marker[α] .= bag

        for i in (last(ν) + 1):n
            marker[i] == bag || push!(η, i)
        end

        Wηα = @view W[η, α]
        Wαα = @view W[α, α]
        Wαν = @view W[α, ν]
        Wην = Wηα * (qr(Wαα, ColumnNorm()) \ Wαν)
        W[η, ν] = Wην
        W[ν, η] = Wην'
        empty!(η)
    end

    return W
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
    if i > j
        i, j = j, i
    end

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
