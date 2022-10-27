"""
    benders_decomposition(inst::TwoStageSpanningTreeInstance;
        MILP_solver=GLPK.Optimizer,
        tol=0.000001,
        silent=false,
    )

Computes an optimal solution (with tolerance tol) of the two stage spanning tree problem
using a Benders approach.
"""
function benders_decomposition(
    inst::TwoStageSpanningTreeInstance;
    MILP_solver=GLPK.Optimizer,
    tol=0.000001,
    silent=false,
)
    model = Model(MILP_solver)
    @variable(model, x[e in edges(inst.g)], Bin)
    @variable(model, theta[i in 1:(inst.nb_scenarios)] >= 0)
    @objective(
        model,
        Min,
        sum(x[e] * inst.first_stage_weights_matrix[src(e), dst(e)] for e in edges(inst.g)) +
            1 / inst.nb_scenarios * sum(theta[scenario] for scenario in 1:(inst.nb_scenarios))
    )

    g = MetaGraph(inst.g)

    scenarios_columns = [[] for _ in 1:(inst.nb_scenarios)]
    previous_scenario = 0

    call_back_counter = 0

    function benders_callback(cb_data)
        if !silent && call_back_counter % 10 == 0
            @info("Benders iteration: $(call_back_counter)")
        end
        call_back_counter += 1

        x_val = callback_value.(cb_data, x)

        for e in edges(g)
            set_prop!(g, e, :x_val, x_val[e])
        end

        for sce in 1:(inst.nb_scenarios)
            scenario = previous_scenario + sce
            if scenario > inst.nb_scenarios
                scenario -= inst.nb_scenarios
            end

            for e in edges(g)
                set_prop!(
                    g, e, :weight, inst.second_stage_weights[scenario][src(e), dst(e)]
                )
            end

            new_feasibility_columns = []
            cut = separate_mst_Benders_cut!(
                g;
                MILP_solver=MILP_solver,
                tol=tol,
                columns=scenarios_columns[scenario],
                new_feasibility_columns=new_feasibility_columns,
            )

            if cut == feasibility_cut
                feasibility_cut_added = @build_constraint(
                    get_prop(g, :nu) + sum(get_prop(g, e, :mu) * x[e] for e in edges(g)) <=
                        0
                )
                MathOptInterface.submit(
                    model, MathOptInterface.LazyConstraint(cb_data), feasibility_cut_added
                )

                previous_scenario = scenario
                return nothing
            end

            if cut == optimality_cut
                theta_val = callback_value(cb_data, theta[scenario])
                lb =
                    sum(
                        x_val[e] * inst.first_stage_weights_matrix[src(e), dst(e)] for
                        e in edges(g)
                    ) + theta_val
                nu = get_prop(g, :nu)
                ub =
                    sum(
                        x_val[e] * inst.first_stage_weights_matrix[src(e), dst(e)] for
                        e in edges(g)
                    ) +
                    nu +
                    sum(
                        (get_prop(g, e, :mu) - get_prop(g, e, :weight)) * x_val[e] for
                        e in edges(g)
                    )
                if ub - lb > tol
                    con = @build_constraint(
                        theta[scenario] >=
                            nu + sum(x[e] * get_prop(g, e, :mu) for e in edges(g)) - sum(
                            x[e] * inst.second_stage_weights[scenario][src(e), dst(e)] for
                            e in edges(g)
                        )
                    )
                    MathOptInterface.submit(
                        model, MathOptInterface.LazyConstraint(cb_data), con
                    )
                    previous_scenario = scenario
                    return nothing
                end
            end
        end
    end

    MathOptInterface.set(model, MathOptInterface.LazyConstraintCallback(), benders_callback)
    optimize!(model)

    return objective_value(model),
    [e for e in edges(g) if abs(value(x[e]) - 1.0) < tol],
    [value(theta[i]) for i in 1:(inst.nb_scenarios)]
end
