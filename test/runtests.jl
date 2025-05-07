using FileIO
using LinearAlgebra
using JuMP
using Test

import MathOptChordalDecomposition
import Mosek
import MosekTools

function construct_model(f, name::String)
    # load data
    data = load("../sdplib/$name.jld2")
    F = data["F"]
    c = data["c"]
    m = data["m"]
    n = data["n"]

    # construct model
    model = JuMP.Model(f)
    set_silent(model)
    @variable(model, x[1:m])
    @objective(model, Min, c' * x)
    @constraint(model, con1, Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m)) in JuMP.PSDCone())
    return model
end

model = construct_model("mcp124-1") do
    MathOptChordalDecomposition.Optimizer(Mosek.Optimizer)
end

JuMP.optimize!(model)
@test round(MOI.get(model.moi_backend.optimizer.model.inner, MOI.ObjectiveValue()); digits = 2) == 141.99

model = construct_model("mcp124-2") do
    MathOptChordalDecomposition.Optimizer(Mosek.Optimizer)
end

JuMP.optimize!(model)
@test round(MOI.get(model.moi_backend.optimizer.model.inner, MOI.ObjectiveValue()); digits = 2) == 269.88

model = construct_model("mcp124-3") do
    MathOptChordalDecomposition.Optimizer(Mosek.Optimizer)
end

JuMP.optimize!(model)
@test round(MOI.get(model.moi_backend.optimizer.model.inner, MOI.ObjectiveValue()); digits = 2) == 467.75

model = construct_model("mcp124-4") do
    MathOptChordalDecomposition.Optimizer(Mosek.Optimizer)
end

JuMP.optimize!(model)
@test round(MOI.get(model.moi_backend.optimizer.model.inner, MOI.ObjectiveValue()); digits = 2) == 864.41
