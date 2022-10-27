"""
    cut_generation(inst::TwoStageSpanningTreeInstance;
        MILP_solver=GLPK.Optimizer,
        separate_constraint_function=separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!,
        tol=0.000001,
        silent=false,
    )

Solve the `TwoStageSpanningTree` instance `inst` using a cut generation approach.

`separate_constraint_function` indicates which method is used to separate the constraint on subsets of vertices. Two options are available
- `separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!`
- `separate_forest_polytope_constraint_vertex_set_using_simple_MILP_formulation!`
"""
function cut_generation(
    inst::TwoStageSpanningTreeInstance;
    MILP_solver=GLPK.Optimizer,
    separate_constraint_function=separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!,
    tol=0.000001,
    silent=false,
)
    model = Model(MILP_solver)
    @variable(model, x[e in edges(inst.g)], Bin)
    @variable(model, y[s in 1:(inst.nb_scenarios), e in edges(inst.g)], Bin)
    @objective(
        model,
        Min,
        sum(x[e] * inst.first_stage_weights_matrix[src(e), dst(e)] for e in edges(inst.g)) +
            1 / inst.nb_scenarios * sum(
            sum(
                y[s, e] * inst.second_stage_weights[s][src(e), dst(e)] for
                s in 1:(inst.nb_scenarios)
            ) for e in edges(inst.g)
        )
    )
    for s in 1:(inst.nb_scenarios)
        @constraint(model, sum(x[e] + y[s, e] for e in edges(inst.g)) == nv(inst.g) - 1)
    end

    g = MetaGraph(inst.g)

    previous_scenario = 0
    call_back_counter = 0

    function sub_graphs_tree_constraint_callback(cb_data)
        if !silent && call_back_counter % 10 == 0
            @info("Cut generated: $(call_back_counter)")
        end
        call_back_counter += 1

        x_val = callback_value.(cb_data, x)
        y_val = callback_value.(cb_data, y)

        for sce in 1:(inst.nb_scenarios)
            scenario = previous_scenario + sce
            if scenario > inst.nb_scenarios
                scenario -= inst.nb_scenarios
            end

            for e in edges(inst.g)
                set_prop!(g, e, :cb_val, x_val[e] + y_val[scenario, e])
            end

            found, X = separate_constraint_function(g; MILP_solver=MILP_solver, tol=tol)

            if found
                con = @build_constraint(
                    sum(
                        x[e] + y[sce, e] for
                        e in edges(inst.g) if (src(e) in X && dst(e) in X)
                    ) <= Int(length(X) - 1)
                )
                MathOptInterface.submit(
                    model, MathOptInterface.LazyConstraint(cb_data), con
                )
                break
            end
        end
    end

    MathOptInterface.set(
        model,
        MathOptInterface.LazyConstraintCallback(),
        sub_graphs_tree_constraint_callback,
    )
    # set_silent(model)
    result = optimize!(model)
    println(result)
    return objective_value(model),
    [e for e in edges(inst.g) if abs(value(x[e]) - 1.0) < tol]
end
