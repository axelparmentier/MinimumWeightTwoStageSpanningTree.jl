"""
    column_generation(inst::TwoStageSpanningTreeInstance;MILP_solver=GLPK.Optimizer,tol=0.00001)

Solves the dual of the linear relaxation of the two stage spanning tree MILP using a constraint generation.

Returns `objective_value`, `duals`
"""
function column_generation(
    inst::TwoStageSpanningTreeInstance; MILP_solver=GLPK.Optimizer, tol=0.00001
)
    model = Model(MILP_solver)

    @variable(model, dummy, Bin) # dummy constraint to activate callbacks

    @variable(model, ν[s in 1:(inst.nb_scenarios)])
    @variable(
        model,
        μ[e in edges(inst.g), s in 1:(inst.nb_scenarios)] <=
            inst.second_stage_weights[s][src(e), dst(e)] / inst.nb_scenarios
    )
    @constraint(
        model,
        first_stage_mu[e in edges(inst.g)],
        sum(μ[e, s] for s in 1:(inst.nb_scenarios)) <=
            inst.first_stage_weights_matrix[src(e), dst(e)]
    )

    @constraint(
        model,
        weak_initial_constraint[s in 1:(inst.nb_scenarios)],
        ν[s] <= sum(μ[e, s] for e in edges(inst.g))
    )

    @objective(model, Max, sum(ν[s] for s in 1:(inst.nb_scenarios)))

    count_call = 0

    function tree_callback(cb_data)
        count_call += 1
        if count_call % 10 == 0
            println(count_call, " callbacks")
        end

        for s in 1:(inst.nb_scenarios)
            weights = deepcopy(inst.second_stage_weights[s])
            for e in edges(inst.g)
                weights[src(e), dst(e)] = callback_value(cb_data, μ[e, s])
                weights[dst(e), src(e)] = weights[dst(e), src(e)]
            end
            mst_val, mst = kruskal_mst_value(inst.g, weights)
            if mst_val + tol < callback_value(cb_data, ν[s])
                con = @build_constraint(ν[s] <= sum(μ[e, s] for e in mst))
                MathOptInterface.submit(
                    model, MathOptInterface.LazyConstraint(cb_data), con
                )
            end
        end
    end

    MathOptInterface.set(model, MathOptInterface.LazyConstraintCallback(), tree_callback)
    optimize!(model)

    println("Nb callbacks: ", count_call)

    θ = zeros(ne(inst.g), inst.nb_scenarios)
    for e in edges(inst.g)
        for s in 1:(inst.nb_scenarios)
            θ[edge_index(inst, e), s] = value(μ[e, s])
        end
    end

    return objective_value(model), θ
end
