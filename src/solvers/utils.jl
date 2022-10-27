@enum BendersSeparationResult feasibility_cut optimality_cut

## Kruskal with solutions

"""
    kruskal_mst_value(g, distmx=weights(g); minimize=true)

extends kruskal_mst to return also the mst value

returns (mst weight), (mst edges)
"""
function kruskal_mst_value(g, distmx=weights(g); kwargs...)
    mst = kruskal_mst(g, distmx; kwargs...)
    sol_value = 0.0
    for e in mst
        sol_value += distmx[src(e), dst(e)]
    end
    return sol_value, mst
end

"""
    separate_mst_Benders_cut!(g::MetaGraph ; MILP_solver=GLPK.Optimizer)

separates optimality and feasibility cuts for the Benders decomposition formulation of the MST problem

input:
property `:x_val` contains the value of the master
property `:weight` contains the value of the second stage

returns a boolean equals to true if the cut is a feasibility cut and false if it is an optimality cut (in that case, the cut might be satisfied)

The value of the duals μ and ν are stored in properties `:mu` and `:nu` of the g

"""
function separate_mst_Benders_cut!(
    g::MetaGraph;
    MILP_solver=GLPK.Optimizer,
    tol=0.000001,
    columns=[],
    new_feasibility_columns=[],
)
    model = Model(MILP_solver)
    @variable(model, 0 <= mu[e in edges(g)] <= 1)
    @variable(model, nu <= 1)
    @variable(model, dummy, Bin)
    @objective(model, Max, nu + sum(mu[e] * get_prop(g, e, :x_val) for e in edges(g)))

    pricing_graph = deepcopy(g)

    function feasibility_tree_constraint_separation_callback(cb_data)
        for e in edges(pricing_graph)
            set_prop!(pricing_graph, e, :weight, callback_value(cb_data, mu[e]))
        end
        tree = kruskal_mst(pricing_graph; minimize=false)
        # constraint_LHS = nu +  sum(mu[e] for e in tree)
        if (
            callback_value(cb_data, nu) +
            sum(callback_value(cb_data, mu[e]) for e in tree) >= tol
        )
            push!(columns, tree)
            push!(new_feasibility_columns, tree)
            con = @build_constraint(nu + sum(mu[e] for e in tree) <= 0)
            MathOptInterface.submit(model, MathOptInterface.LazyConstraint(cb_data), con)
        end
    end
    set_silent(model)
    MathOptInterface.set(
        model,
        MathOptInterface.LazyConstraintCallback(),
        feasibility_tree_constraint_separation_callback,
    )
    optimize!(model)

    if (objective_value(model) > tol)
        # Feasibility constraint priced
        for e in edges(g)
            set_prop!(g, e, :mu, value.(mu[e]))
        end
        set_prop!(g, :nu, value(nu))
        return feasibility_cut
    end

    model = Model(MILP_solver)
    @variable(model, 0 <= mu[e in edges(g)])
    @variable(model, nu)
    @variable(model, dummy, Bin)
    @objective(model, Max, nu + sum(mu[e] * get_prop(g, e, :x_val) for e in edges(g)))

    for tree in columns
        @constraint(
            model,
            nu + sum(mu[e] for e in tree) - sum(get_prop(g, e, :weight) for e in tree) <= 0
        )
    end

    function optimality_tree_constraint_separation_callback(cb_data)
        for e in edges(pricing_graph)
            set_prop!(
                pricing_graph,
                e,
                :weight,
                callback_value(cb_data, mu[e]) - get_prop(g, e, :weight),
            )
        end
        tree = kruskal_mst(pricing_graph; minimize=false)
        if (
            callback_value(cb_data, nu) +
            sum(callback_value(cb_data, mu[e]) for e in tree) -
            sum(get_prop(g, e, :weight) for e in tree) >= tol
        )
            push!(columns, tree)

            con = @build_constraint(
                nu + sum(mu[e] for e in tree) -
                sum(get_prop(g, e, :weight) for e in tree) <= 0
            )
            MathOptInterface.submit(model, MathOptInterface.LazyConstraint(cb_data), con)
        end
    end
    set_silent(model)
    MathOptInterface.set(
        model,
        MathOptInterface.LazyConstraintCallback(),
        optimality_tree_constraint_separation_callback,
    )
    optimize!(model)

    for e in edges(g)
        set_prop!(g, e, :mu, value.(mu[e]))
    end
    set_prop!(g, :nu, value(nu))

    return optimality_cut
end

## Separate constraints based on min cuts MILP

"""
    separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!(g::MetaGraph; MILP_solver=GLPK.Optimizer)

Uses a simple MILP to separate the constraints
Constraint separated: ∑_{e in E(X)} x_e <= |X| - 1 for any `∅ ⊊ X ⊊ V``

- g is a MetaGraph, `:cb_val` property contains the value of x_e

returns: found, X
- found : boolean indicating if a violated constraint has been found
- X : vertex set corresponding to the violated constraint

"""
function separate_forest_polytope_constraint_vertex_set_using_min_cut_MILP_formulation!(
    g::MetaGraph; MILP_solver=GLPK.Optimizer, tol=0.000001, infinity_cst=1000.0
)
    # build flow graph
    flow_graph = MetaDiGraph()
    base_graph_edges_vertex_index_in_flow_graph = Dict()
    base_graph_vertices_vertex_index_in_flow_graph = Dict()
    s, t = build_flow_graph_for_constraint_pricing!(
        g,
        flow_graph,
        base_graph_vertices_vertex_index_in_flow_graph,
        base_graph_edges_vertex_index_in_flow_graph,
    )

    # build MILP model
    model = Model(MILP_solver)

    @variable(model, y[v in vertices(flow_graph)], Bin)
    @variable(model, z[e in edges(flow_graph)], Bin)

    @objective(
        model,
        Min,
        sum(
            z[e] * min(infinity_cst, get_prop(flow_graph, e, :ub)) for
            e in edges(flow_graph)
        )
    )

    @constraint(model, origin_destination, y[s] - y[t] >= 1)
    @constraint(
        model, edge_constraint[e in edges(flow_graph)], z[e] >= y[src(e)] - y[dst(e)]
    )
    @constraint(
        model,
        non_empty_X,
        sum(y[base_graph_vertices_vertex_index_in_flow_graph[v]] for v in vertices(g)) >= 1
    )

    # solve
    set_silent(model)
    optimize!(model)
    min_cut_value = objective_value(model)

    # not cut priced ?

    if min_cut_value >= nv(g) - tol
        return false, []
    end

    # rebuild
    return true,
    [
        v for v in vertices(g) if
        value(y[base_graph_vertices_vertex_index_in_flow_graph[v]]) >= 1 - tol
    ]
end

### DiGraph used

"""
    function build_flow_graph_for_constraint_pricing!(
        g::MetaGraph,
        flow_graph::MetaDiGraph,
        base_graph_vertices_vertex_index_in_flow_graph = Dict(),
        base_graph_edges_vertex_index_in_flow_graph = Dict()
    )

Constraint separated: ∑_{e in E(X)}x_e <= |X| - 1 for any `∅ ⊊ X ⊊ V``
- g is the undirected graph on which the forest polytope is manipulated.
- flow_graph is a MetaDiGraph. It should be empty in input. It is modified by the function and contains afterwards the MetaDiGraph used to separate the forest polytope constraint on f
"""
function build_flow_graph_for_constraint_pricing!(
    g::AbstractGraph,
    flow_graph::MetaDiGraph,
    base_graph_vertices_vertex_index_in_flow_graph=Dict(),
    base_graph_edges_vertex_index_in_flow_graph=Dict(),
)
    ## Build vertices
    # Origin and destination
    add_vertices!(flow_graph, 2)
    s = 1
    t = 2

    # flow_graph vertices that correspond to vertices of base graph
    for e in edges(g)
        add_vertex!(flow_graph)
        base_graph_edges_vertex_index_in_flow_graph[e] = nv(flow_graph)
    end

    # flow_graph vertices that correspond to edges of base graph
    for v in vertices(g)
        add_vertex!(flow_graph)
        base_graph_vertices_vertex_index_in_flow_graph[v] = nv(flow_graph)
    end

    ## Build edges
    for e in edges(g)
        v_edge = base_graph_edges_vertex_index_in_flow_graph[e]
        v_srv = base_graph_vertices_vertex_index_in_flow_graph[src(e)]
        v_dst = base_graph_vertices_vertex_index_in_flow_graph[dst(e)]

        add_edge!(flow_graph, Edge(s, v_edge))
        set_prop!(flow_graph, Edge(s, v_edge), :ub, get_prop(g, e, :cb_val))

        for w in [v_srv, v_dst]
            add_edge!(flow_graph, Edge(v_edge, w))
            set_prop!(flow_graph, Edge(v_edge, w), :ub, Inf)
        end
    end

    for v in vertices(g)
        v_vert = base_graph_vertices_vertex_index_in_flow_graph[v]
        add_edge!(flow_graph, Edge(v_vert, t))
        set_prop!(flow_graph, Edge(v_vert, t), :ub, 1)
    end
    return s, t
end
