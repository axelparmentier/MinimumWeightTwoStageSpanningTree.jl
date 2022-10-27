function first_stage_optimal_solution(
    inst::TwoStageSpanningTreeInstance, θ::AbstractArray; box_size_for_x=20.0
)
    edge_weight_vector =
        inst.first_stage_weights_vector +
        dropdims(1 / inst.nb_scenarios * sum(θ; dims=2); dims=2)

    edges_index_with_negative_cost = Int[
        e_ind for e_ind in 1:ne(inst.g) if edge_weight_vector[e_ind] < 0
    ]
    value = 0.0
    if length(edges_index_with_negative_cost) > 0
        value = sum(
            box_size_for_x * edge_weight_vector[e_ind] for
            e_ind in edges_index_with_negative_cost
        )
    end

    grad = zeros(ne(inst.g), inst.nb_scenarios)
    for e_ind in edges_index_with_negative_cost
        for i in 1:(inst.nb_scenarios)
            grad[e_ind, i] = box_size_for_x * 1 / inst.nb_scenarios
        end
    end
    return value, grad
end

function second_stage_optimal_solution(
    inst::TwoStageSpanningTreeInstance,
    θ::AbstractArray,
    scenario::Integer,
    grad::AbstractArray,
)
    weights = deepcopy(inst.second_stage_weights[scenario])
    for e in edges(inst.g)
        e_ind = edge_index(inst, e)
        w = -θ[e_ind, scenario]
        if w < weights[src(e), dst(e)]
            weights[src(e), dst(e)] = w
            weights[dst(e), src(e)] = w
        end
    end

    value, tree = kruskal_mst_value(inst.g, weights)

    for e in tree
        e_ind = edge_index(inst, e)
        w = -θ[e_ind, scenario]
        if w < inst.second_stage_weights[scenario][src(e), dst(e)]
            grad[e_ind, scenario] += -1 / inst.nb_scenarios
        end
    end
    return 1 / inst.nb_scenarios * value
end

function lagrangian_function_value_gradient(
    inst::TwoStageSpanningTreeInstance, θ::AbstractArray
)
    value, grad = first_stage_optimal_solution(inst, θ)
    # grads = zeros(ne(inst.g),inst.nb_scenarios,inst.nb_scenarios)
    values = zeros(inst.nb_scenarios)
    Threads.@threads for i in 1:(inst.nb_scenarios)
        # for i in 1:inst.nb_scenarios
        values[i] = second_stage_optimal_solution(inst, θ, i, grad) # Different part of grad are modified
    end
    value += sum(values)
    # grad += dropdims(sum(grads,dims=3),dims=3)
    return value, grad
end

"""
    lagrangian_heuristic(θ::AbstractArray; inst::TwoStageSpanningTreeInstance)

Performs a lagrangian heuristic on TwoStageSpanningTree instance inst with duals θ.

θ[edge_index(src(e),dst(e)),s] contains the value of the Lagrangian dual corresponding to edge e for scenario s.

Return (value of the solution computed),(edges in the solution computed).
"""
function lagrangian_heuristic(θ::AbstractArray; inst::TwoStageSpanningTreeInstance)
    # Compute - x_{es} using grad
    grad = zeros(ne(inst.g), inst.nb_scenarios)
    Threads.@threads for i in 1:(inst.nb_scenarios)
        second_stage_optimal_solution(inst, θ, i, grad)
    end
    # Compute the average x_{es} and build a graph that is a candidate spannning tree (but not necessarily a spanning tree nor a forest)
    average_x = -dropdims(sum(grad; dims=2); dims=2)
    candidate_solution_indicator = round.(average_x)
    candidate_first_stage_solution = [
        e for e in edges(inst.g) if
        abs(1 - candidate_solution_indicator[edge_index(inst, e)]) < 0.00001
    ]
    # Build a spanning tree that contains as many edges of our candidate as possible
    I = Int[]
    J = Int[]
    V = Int[]
    for e in candidate_first_stage_solution
        push_edge_in_weight_matrix!(I, J, V, e, 1)
    end
    weights = sparse(I, J, V, nv(inst.g), nv(inst.g))
    tree_from_candidate = kruskal_mst(inst.g, weights; minimize=false)
    # Keep only the edges that are in the initial candidate graph and in the spanning tree
    forest = intersect(tree_from_candidate, candidate_first_stage_solution)
    return evaluate_first_stage_solution(inst, forest), forest
    # TODO: look if we can improve by taking, instead of the indicator, the average values in the MST weights
end

"""
    lagrangian_dual(θ::AbstractArray; inst::TwoStageSpanningTreeInstance)

Compute the value of the lagrangian dual function for TwoStageSpanningTreeInstance instance inst with duals θ

`θ[edged_index(src(e),dst(e)),s]` contains the value of the Lagrangian dual corresponding to edge e for scenario s

ChainRulesCore.rrule automatic differentiation with respect to θ works.

Return (value of the solution computed),(edges in the solution computed)
"""
function lagrangian_dual(θ::AbstractArray; inst::TwoStageSpanningTreeInstance)
    value, _ = lagrangian_function_value_gradient(inst, θ)
    return value
end

"""
    lagrangian_relaxation(inst::TwoStageSpanningTreeInstance; nb_epochs=100)

Runs
- a subgradient algorithm during nb_epochs to solve the Lagrangian relaxation of inst (with `θ` initialized to 0)
- a lagrangian heuristic on the resulting solution

Returns `lb`, `ub`, `forest`, `θ`, `training_losses` with
- `lb`: value of the Lagrangian relaxation
- `ub`: value of the solution computed by the lagrangian heuristic
- `forest`: solution computed by the lagrangian heuristic
- `θ`: second stage problem value
- `training_losses`: vector with the value of the training losses along the subgradient algorithm
"""
function lagrangian_relaxation(
    inst::TwoStageSpanningTreeInstance; nb_epochs=100, stop_gap=0.00001, show_progress=true
)
    θ = zeros(ne(inst.g), inst.nb_scenarios)
    opt = Adam()

    lb = -Inf
    best_theta = θ
    ub = Inf
    forest = Edge{Int}[]
    last_ub_epoch = -1000

    training_losses = Float64[]
    p = Progress(nb_epochs; enabled=show_progress)
    for epoch in 1:nb_epochs
        value, grad = lagrangian_function_value_gradient(inst, θ)

        if value > lb
            lb = value
            best_theta = θ
            if epoch > last_ub_epoch + 100
                last_ub_epoch = epoch
                ub, forest = lagrangian_heuristic(θ; inst=inst)
                if lb > 0.0 && (ub - lb) / abs(lb) <= stop_gap
                    println("Stopped after ", epoch, " gap smaller than ", stop_gap)
                    break
                end
            end
        end
        Flux.update!(opt, θ, -grad)
        push!(training_losses, value)
        next!(p)
    end

    θ = best_theta

    return lb, ub, forest, θ, training_losses
end
