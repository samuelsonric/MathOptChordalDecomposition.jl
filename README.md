# MathOptChordalDecomposition.jl

MathOptChordalDecomposition.jl is a [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) layer that implements chordal decomposition of
sparse semidefinite constraints.

## Basic Usage

The `sdplib` directory contains three semidefinite programming problems from the [SDPLIB library](https://github.com/vsdp/SDPLIB). The function `construct_model`, defined below, reads one of the problems and constructs a [JuMP.jl](https://github.com/jump-dev/JuMP.jl) model.

```julia-repl
julia> using FileIO, LinearAlgebra, JuMP

julia> import MathOptChordalDecomposition, MosekTools, Mosek

julia> function construct_model(f, name::String)
           # load data
           data = load("./sdplib/$name.jld2");
           F = data["F"]
           c = data["c"]
           m = data["m"]
           n = data["n"]
    
           # construct model
           model = JuMP.Model(f)
           set_silent(model)
           @variable(model, x[1:m])
           @objective(model, Min, c' * x)
           @constraint(model, con1,  Symmetric(-Matrix(F[1]) + sum(Matrix(F[k + 1]) .* x[k] for k in 1:m)) in JuMP.PSDCone())
           return model
       end
construct_model (generic function with 1 method)
```

Solve the problem using the [Mosek.jl](https://github.com/MOSEK/Mosek.jl) optimizer.

```julia-repl
julia> model = construct_model(Mosek.Optimizer, "mcp124-1");

julia> @time JuMP.optimize!(model)
  6.005076 seconds (2.53 k allocations: 515.859 KiB)

julia> MOI.get(model, MOI.ObjectiveValue())
141.9904770422396
```

Solve the problem using [Mosek.jl](https://github.com/MOSEK/Mosek.jl) and MathOptChordalDecomposition.jl.

```julia-repl
julia> model = construct_model("mcp124-1") do
           MathOptChordalDecomposition.Optimizer(Mosek.Optimizer)
       end;

julia> @time JuMP.optimize!(model)
  0.041175 seconds (230.72 k allocations: 11.800 MiB)

julia> MOI.get(model.moi_backend.optimizer.model.inner, MOI.ObjectiveValue())
141.99047611570586
```
