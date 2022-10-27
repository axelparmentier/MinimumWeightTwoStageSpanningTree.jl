
"""
    kruskal_maximum_weight_forest(θ::AbstractVector;inst=inst)

Return a maximum weight forest on `inst.g` using θ as edge weight (returns a vector of edges)
"""
function kruskal_maximum_weight_forest(
    edge_weights_vector::AbstractVector, inst::TwoStageSpanningTreeInstance
)
    weights = deepcopy(inst.first_stage_weights_matrix)
    for e in edges(inst.g)
        e_ind = edge_index(inst, e)
        w = edge_weights_vector[e_ind]
        weights[src(e), dst(e)] = w
        weights[dst(e), src(e)] = w
    end

    mst = kruskal_mst(inst.g, weights; minimize=false)
    forest = [e for e in mst if edge_weights_vector[edge_index(inst, e)] > 0]

    return forest
end

"""
    maximum_weight_forest_linear_maximizer(θ::AbstractVector;inst=inst)

Wrapper around kruskal_maximum_weight_forest(edge_weights_vector::AbstractVector, inst::TwoStageSpanningTreeInstance) that returns the solution encoded as a vector.
"""
function maximum_weight_forest_linear_maximizer(θ::AbstractVector; inst=inst)
    forest = kruskal_maximum_weight_forest(θ, inst)
    y = zeros(ne(inst.g))
    for e in forest
        y[edge_index(inst, e)] = 1.0
    end
    return y
end

"""
    maximum_weight_forest_layer_linear_encoder(inst::TwoStageSpanningTreeInstance)

Returns X::Array{Float64} with `X[f,edge_index(inst,e),s]` containing the value of feature number `f` for edge `e` and scenario `s`

# Features used: (all are homogeneous to a cost)
- `first_stage_cost`
- `second_stage_cost_quantile`
- `best_stage_cost_quantile`
- `neighbors_first_stage_cost_quantile`
- `neighbors_scenario_second_stage_cost_quantile`
- `is_in_first_stage_x_first_stage_cost`
- `is_in_second_stage_x_second_stage_cost_quantile`
- `is_first_in_best_stage_x_best_stage_cost_quantile`
- `is_second_in_best_stage_x_best_stage_cost_quantile`

For features with quantiles, the following quantiles are used: 0:0.1:1.
"""
function maximum_weight_forest_layer_linear_encoder(inst::TwoStageSpanningTreeInstance)

    # Choose quantiles
    quantiles_used = [i for i in 0.0:0.1:1.0]

    # MST features
    fs_mst_indicator = compute_first_stage_mst(inst)
    ss_mst_indicator = compute_second_stage_mst(inst)
    bfs_mst_indicator, bss_mst_indicator = compute_best_stage_mst(inst)
    second_stage_edge_costs = pivot_instance_second_stage_costs(inst)
    edge_neighbors = compute_edge_neighbors(inst)

    # Build features
    nb_features = 2 + 7 * length(quantiles_used)
    X = zeros(Float64, nb_features, ne(inst.g))

    for e in edges(inst.g)
        count_feat = 0
        e_ind = edge_index(inst, e)

        function add_quantile_features(realizations)
            for p in quantiles_used
                count_feat += 1
                X[count_feat, e_ind] = quantile(realizations, p)
            end
        end

        ## Costs features
        # first_stage_cost
        count_feat += 1
        X[count_feat, e_ind] = inst.first_stage_weights_vector[e_ind]

        # second_stage_cost_quantile
        edge_second_stage_costs = [
            inst.second_stage_weights[s][src(e), dst(e)] for s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_second_stage_costs)

        # best_stage_cost_quantile
        edge_best_stage_costs = [
            min(
                inst.first_stage_weights_matrix[src(e), dst(e)],
                inst.second_stage_weights[s][src(e), dst(e)],
            ) for s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_best_stage_costs)

        ## Neighbors features
        # neighbors_first_stage_cost_quantile,
        edge_neighbors_first_stage_cost_quantile = [
            inst.first_stage_weights_vector[e_i] for e_i in edge_neighbors[e_ind]
        ]
        add_quantile_features(edge_neighbors_first_stage_cost_quantile)

        # neighbors_scenario_second_stage_cost_quantile
        edge_neighbors_scenario_second_stage_cost_quantile = [
            second_stage_edge_costs[n, s] for s in 1:(inst.nb_scenarios) for
            n in edge_neighbors[e_ind]
        ]
        add_quantile_features(edge_neighbors_scenario_second_stage_cost_quantile)

        ## MST features
        # is_in_first_stage_x_first_stage_cost
        count_feat += 1
        X[count_feat, e_ind] =
            fs_mst_indicator[e_ind] * inst.first_stage_weights_vector[e_ind]

        # is_in_second_stage_x_second_stage_cost_quantile
        edge_is_in_second_stage_x_second_stage_cost_quantile = [
            ss_mst_indicator[e_ind, s] * second_stage_edge_costs[e_ind, s] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_in_second_stage_x_second_stage_cost_quantile)

        # is_first_in_best_stage_x_best_stage_cost_quantile
        edge_is_first_in_best_stage_x_best_stage_cost_quantile = [
            bfs_mst_indicator[e_ind, s] * inst.first_stage_weights_vector[e_ind] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_first_in_best_stage_x_best_stage_cost_quantile)

        # is_second_in_best_stage_x_best_stage_cost_quantile
        edge_is_second_in_best_stage_x_best_stage_cost_quantile = [
            bss_mst_indicator[e_ind, s] * second_stage_edge_costs[e_ind, s] for
            s in 1:(inst.nb_scenarios)
        ]
        add_quantile_features(edge_is_second_in_best_stage_x_best_stage_cost_quantile)
    end
    return X
end

"""
    build_solve_and_encode_instance_as_maximum_weight_forest(;
        grid_size=3,
        seed=0,
        nb_scenarios=10,
        first_max=10,
        second_max=20,
        solver=lagrangian_heuristic_solver,
        load_and_save=true
    )

Builds a two stage spanning tree instance with a square grid graph of width `grid_size`, `seed` for the random number generator, `nb_scenarios` for the second stage, `first_max` and `second_max` as first and second stage maximum weight.

Solves it with `solver`.

Encodes it for a pipeline with a maximum weight forest layer.
"""
function build_solve_and_encode_instance_as_maximum_weight_forest(;
    grid_size=3,
    seed=0,
    nb_scenarios=10,
    first_max=10,
    second_max=20,
    solver=lagrangian_heuristic_solver,
    load_and_save=true,
)
    inst, lb, ub, sol = build_load_or_solve(;
        grid_size=grid_size,
        seed=seed,
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        solver=solver,
        load_and_save=load_and_save,
    )

    x = maximum_weight_forest_layer_linear_encoder(inst)

    y = zeros(ne(inst.g))
    for e in sol
        y[edge_index(inst, e)] = 1.0
    end

    return x, y, (inst=inst, lb=lb, ub=ub, sol=sol)
end
