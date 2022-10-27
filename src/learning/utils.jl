solver_name(solv::Any) = "unknown"

function cut_solver(inst::TwoStageSpanningTreeInstance)
    lb, sol = cut_generation(inst; silent=true)
    ub = lb
    return lb, ub, sol
end

solver_name(solv::typeof(cut_solver)) = "cut"

function benders_solver(inst::TwoStageSpanningTreeInstance)
    lb, sol, _ = benders_decomposition(inst; silent=true)
    ub = lb
    return lb, ub, sol
end

solver_name(solv::typeof(benders_solver)) = "benders"

# function lagrangian_heuristic_solver(inst::TwoStageSpanningTreeInstance)
#     lb,ub,sol,_,_ = two_stage_spanning_tree_lagrangian_relaxation(inst,nb_epochs=20000,show_progress=false)
#     return lb,ub,sol
# end

struct LagrangianHeuristicSolver
    max_iter::Int
    stop_gap::Float64
end

function (solv::LagrangianHeuristicSolver)(inst::TwoStageSpanningTreeInstance)
    lb, ub, sol, _, _ = lagrangian_relaxation(
        inst; nb_epochs=solv.max_iter, stop_gap=solv.stop_gap, show_progress=true
    )
    return lb, ub, sol
end

lagrangian_heuristic_solver = LagrangianHeuristicSolver(20000, 0.001)

function solver_name(solv::LagrangianHeuristicSolver)
    return "lagrangian-it" * string(solv.max_iter) * "-gap" * string(solv.stop_gap)
end

"""
    build_load_or_solve(;
        graph=grid(5,5),
        seed=0,
        nb_scenarios=10,
        first_max=10,
        second_max=20,
        solver=lagrangian_heuristic_solver,
        load_and_save=true
    )

# Three solvers available
- `cut_solver`
- `benders_solver`
- `lagrangian_heuristic_solver`

Return `inst`, `val`, `sol` with:
- `inst`: the instance generated
- `val`: value of the solution computed
- `solution_computed`: solution computed

"""
function build_load_or_solve(;
    grid_size=3,
    seed=0,
    nb_scenarios=10,
    first_max=10,
    second_max=20,
    solver=lagrangian_heuristic_solver,
    load_and_save=true,
)
    Random.seed!(seed)

    graph = grid((grid_size, grid_size))

    inst = TwoStageSpanningTreeInstance(
        graph; nb_scenarios=nb_scenarios, first_max=first_max, second_max=second_max
    )

    #val = 0.0
    lb = -1.0
    ub = -1.0

    if load_and_save
        inst_sol_name =
            "sol_" *
            solver_name(solver) *
            "_gs" *
            string(grid_size) *
            "_seed" *
            string(seed) *
            "_sce" *
            string(nb_scenarios) *
            "_fm" *
            string(first_max) *
            "_sm" *
            string(second_max) *
            ".json"
        mkpath("data")
        filename = joinpath("data", inst_sol_name)

        if isfile(filename)
            # load
            stringdata = ""
            f = open(filename, "r")
            stringdata = read(f, String)
            close(f)
            sol_dict = JSON.parse(stringdata)
            lb = sol_dict[1]
            ub = sol_dict[2]
            sol = Vector{Edge}(undef, length(sol_dict[3]))
            for (i, d) in enumerate(sol_dict[3])
                sol[i] = Edge(d["src"], d["dst"])
            end
        else
            lb, ub, sol = solver(inst)
            # Save
            stringdata = JSON.json((lb, ub, sol))
            open(filename, "w") do f
                write(f, stringdata)
            end
        end
        #val = evaluate_first_stage_solution(inst, sol)
    else
        lb, ub, sol = solver(inst; MILP_solver=MILP_solver)
    end

    return inst, lb, ub, sol
end

# """
#     BlackBoxLoss{A<:AbstractArray,I,P} <: Function

# Encodes all the information for learning a problem by experience.
# """
# struct BlackBoxLoss{A<:AbstractArray,I,P} <: Function
#     nb_features::Int
#     nb_samples::Int
#     training_set::Vector{Tuple{A,I,Float64}}
#     cost_pipeline::P     # Cost ∘ Decoder ∘ Maximizer: Outputs a solution
# end

# """
#     function BlackBoxLoss(training_data,maximizer,decoder,cost_function;scaling_function=x->1.0)

# Constructor for a `BlackBoxLoss`

# Pipeline: `x` --GLM--> `θ` --Maximizer--> `y` --Decoder--> `z`

#  - `training_data`: Vector(`x`,`inst`) where `inst` is an instance of the problem and `x` its features encoding
#  - `maximizer(θ,inst)`
#  - `decoder(y,inst)`
#  - `cost_function(z,inst)`
#  - `scaling_function`: order of magnitude of `cost_function(z_optimal,inst)`:
# """
# function BlackBoxLoss(
#     training_data,
#     maximizer,
#     decoder,
#     cost_function;
#     scaling_function = x -> 1.0,
# )
#     nb_features = size(training_data[1][1])[1]
#     nb_samples = length(training_data)
#     training_set = [(x, inst, scaling_function(inst)) for (x, inst) in training_data]
#     cost_pipeline = (θ, inst) -> cost_function(decoder(maximizer(θ, inst), inst), inst)
#     return BlackBoxLoss(nb_features, nb_samples, training_set, cost_pipeline)
# end

# """
#     function black_box_loss(w::AbstractVector,bbl::BlackBoxLoss)

# Evaluates `BlackBoxLoss` when its GLM parameters are given by `w`
# """
# function black_box_loss(w::AbstractVector, bbl::BlackBoxLoss)
#     model = Chain(Dense(w', false), x -> dropdims(x; dims = 1))
#     return ThreadsX.sum(
#         1 / u * bbl.cost_pipeline(model(x), inst) for (x, inst, u) in bbl.training_set
#     )
# end
