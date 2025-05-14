# MathOptChordalDecomposition.jl

[![CI](https://github.com/samuelsonric/MathOptChordalDecomposition.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelsonric/MathOptChordalDecomposition.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/samuelsonric/MathOptChordalDecomposition.jl/graph/badge.svg?token=z67ISx3vkD)](https://codecov.io/gh/samuelsonric/MathOptChordalDecomposition.jl)

MathOptChordalDecomposition.jl is a [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl)
layer that implements chordal decomposition of sparse semidefinite constraints.

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/samuelsonric/MathOptChordalDecomposition.jl/issues/new).

## License

`MathOptChordalDecomposition.jl` is licensed under the
[MIT License](https://github.com/samuelsonric/MathOptChordalDecomposition.jl/blob/master/LICENSE).

## Installation

Install MathOptChordalDecomposition as follows:
```julia
import Pkg
Pkg.add("MathOptChordalDecomposition")
```

## Use with JuMP

To use MathOptChordalDecomposition with JuMP, use `MathOptChordalDecomposition.Optimizer`:

```julia
using JuMP, MathOptChordalDecomposition, SCS
model = Model(() -> MathOptChordalDecomposition.Optimizer(SCS.Optimizer))
```
Change `SCS` for any other conic solver that supports semidefinite constraints.

## Basic Usage

The `sdplib` directory contains four semidefinite programming problems from the
[SDPLIB library](https://github.com/vsdp/SDPLIB).

The function `construct_model`, defined below, reads one of the problems and
constructs a [JuMP.jl](https://github.com/jump-dev/JuMP.jl) model.

For this example, it is significantly faster to solve the problem with 
MathOptChordalDecomposition than to use SCS by itself:

```julia
julia> using FileIO, JLD2, JuMP, LinearAlgebra, SCS

julia> import MathOptChordalDecomposition as MOCD

julia> function main(optimizer, name::String)
           data = FileIO.load("./sdplib/$name.jld2");
           F, c, m, n = data["F"], data["c"], data["m"], data["n"]    
           model = Model(optimizer)
           set_silent(model)
           @variable(model, x[1:m])
           @objective(model, Min, c' * x)
           @constraint(
               model,
               con,
               LinearAlgebra.Symmetric(-F[1] + x' * F[2:end]) in PSDCone(),
           )
           optimize!(model)
           return objective_value(model)
       end
main (generic function with 1 method)

julia> @time main(SCS.Optimizer, "mcp124-1")
  9.447474 seconds (154.70 k allocations: 12.313 MiB)
141.96561765120785

julia> @time main(() -> MOCD.Optimizer(SCS.Optimizer), "mcp124-1")
  0.245992 seconds (170.72 k allocations: 15.103 MiB, 1.85% compilation time)
141.9887372030578
```
