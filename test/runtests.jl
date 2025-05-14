using FileIO
using LinearAlgebra
using JuMP
using Test

import MathOptChordalDecomposition as MOCD
import MathOptInterface as MOI
import SCS

# ------ #
# SDPLib #
# ------ #

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
    @constraint(
        model,
        con,
        Symmetric(-Matrix(F[1]) + sum(Matrix(F[k+1]) .* x[k] for k in 1:m)) in
        JuMP.PSDCone()
    )
    return model, con
end

for name in ("mcp124-1", "mcp124-2", "mcp124-3", "mcp124-4")
    old, oldcon = construct_model(SCS.Optimizer, name)
    new, newcon = construct_model(() -> MOCD.Optimizer(SCS.Optimizer), name)

    JuMP.optimize!(old)
    JuMP.optimize!(new)

    # primal
    @test -1 < objective_value(old) - objective_value(new) < 1
    @test -1 < norm(value(oldcon) - value(newcon)) < 1

    # dual
    @test -1 < dual_objective_value(old) - dual_objective_value(new) < 1
    #@test -1 < norm(dual(oldcon) - dual(newcon)) < 1
end

# ---------------- #
# MathOptInterface #
# ---------------- #

model = MOI.instantiate(
    () -> MOCD.Optimizer(SCS.Optimizer);
    with_bridge_type = Float64,
    with_cache_type = Float64,
)

MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1e-6)

MOI.Test.runtests(
    model,
    MOI.Test.Config(;
        atol = 1e-3,
        rtol = 1e-3,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    );
    exclude = String[
        "test_linear_add_constraints",
        "test_conic_HermitianPositiveSemidefiniteConeTriangle_2",
        "test_model_ModelFilter_AbstractConstraintAttribute",
        "test_attribute_RawStatusString",
        "test_attribute_SolveTimeSec",
        "test_objective_ObjectiveFunction_blank",
        "test_solve_TerminationStatus_DUAL_INFEASIBLE",
    ],
)
