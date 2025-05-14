using FileIO
using LinearAlgebra
using JuMP
using Test

import MathOptChordalDecomposition as MOCD
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
    @constraint(model, con, Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m)) in JuMP.PSDCone())
    return model, con
end

for name in ("mcp124-1", "mcp124-2", "mcp124-3", "mcp124-4")
    old, oldcon = construct_model(Mosek.Optimizer, name)
    new, newcon = construct_model(() -> MOCD.Optimizer(Mosek.Optimizer), name)

    JuMP.optimize!(old)
    JuMP.optimize!(new)

    # primal
    @test -0.005 < objective_value(old) - objective_value(new) < 0.005
    @test -0.005 < norm(value(oldcon) - value(newcon)) < 0.005

    # dual
    @test -0.005 < dual_objective_value(old) - dual_objective_value(new) < 0.005
    @test -0.005 < norm(dual(oldcon) - dual(newcon)) < 0.005
end
