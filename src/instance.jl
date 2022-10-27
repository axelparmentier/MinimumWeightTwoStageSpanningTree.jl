"""
    TwoStageSpanningTreeInstance

Contains all the relevant information defining a two stage spanning-tree instance.

# Fields
- `g::SimpleGraph{Int}`: the graph
- `edge_index::SparseMatrixCSC{Int64, Int64}`: edge_index[src(e),dst(e)] contains the index of edge `e`
- `nb_scenarios::Int`: number of scenarios
- `first_stage_weights_matrix::SparseMatrixCSC{Float64, Int64}`:
- `first_stage_weights_vector::Vector{Float64}`:
- `second_stage_weights::Vector{SparseMatrixCSC{Float64, Int64}}`:
"""
struct TwoStageSpanningTreeInstance
    g::SimpleGraph{Int}
    edge_index::SparseMatrixCSC{Int64,Int64}
    nb_scenarios::Int
    first_stage_weights_matrix::SparseMatrixCSC{Float64,Int64}
    first_stage_weights_vector::Vector{Float64}
    second_stage_weights::Vector{SparseMatrixCSC{Float64,Int64}}
end

"""
    function edge_index(inst::TwoStageSpanningTreeInstance, e::AbstractEdge)

Returns `inst.edge_index[src(e),dst(e)]`.
"""
function edge_index(inst::TwoStageSpanningTreeInstance, e::AbstractEdge)
    return inst.edge_index[src(e), dst(e)]
end

function push_edge_in_weight_matrix!(
    I::Vector{Int}, J::Vector{Int}, V::Vector{T}, e::AbstractEdge, w::T
) where {T<:Real}
    push!(I, src(e))
    push!(J, dst(e))
    push!(V, w)
    push!(I, dst(e))
    push!(J, src(e))
    return push!(V, w)
end

function build_edge_indices(g::AbstractGraph)
    e_ind = 0
    I = Int[]
    J = Int[]
    V = Int[]
    for e in edges(g)
        e_ind += 1
        push_edge_in_weight_matrix!(I, J, V, e, e_ind)
    end
    return sparse(I, J, V, nv(g), nv(g))
end

function get_weight_matrix_from_weight_vector(
    g::AbstractGraph, edge_index::AbstractMatrix, weightVector::Vector{T}
) where {T}
    I = Int[]
    J = Int[]
    V = T[]
    for e in edges(g)
        e_ind = edge_index[src(e), dst(e)]
        push_edge_in_weight_matrix!(I, J, V, e, weightVector[e_ind])
    end
    return sparse(I, J, V, nv(g), nv(g))
end

function get_weight_vector_from_weight_matrix(
    g::AbstractGraph, edge_index::AbstractMatrix, weight_matrix::SparseMatrixCSC{T,Ti}
) where {T,Ti<:Integer}
    result = Vector{T}(undef, ne(g))
    for e in edges(g)
        result[edge_index[src(e), dst(e)]] = weight_matrix[src(e), dst(e)]
    end
    return result
end

function random_weights(g::AbstractGraph; w_max=20)
    weights = spzeros(nv(g), nv(g))
    for e in edges(g)
        weights[src(e), dst(e)] = abs(rand(Int, 1)[1]) % w_max
        weights[dst(e), src(e)] = weights[src(e), dst(e)]
    end
    return weights
end

"""
    TwoStageSpanningTreeInstance(g::AbstractGraph;nb_scenarios=1, first_max=10, second_max=20)

Build a random TwoStageSpanningTreeInstance from graph `g`.

`nb_scenarios` scenarios are drawn, with first-stage weights drawn uniformly between 0 and
`first_max`, and second-stage weights drawn uniformly between 0 an d`second_max`.
"""
function TwoStageSpanningTreeInstance(
    g::AbstractGraph; nb_scenarios=1, first_max=10, second_max=20
)
    edge_ind = build_edge_indices(g)
    first_stage_weights_matrix = random_weights(g; w_max=first_max)
    first_stage_weights_vector = get_weight_vector_from_weight_matrix(
        g, edge_ind, first_stage_weights_matrix
    )
    second_stage_weights = [random_weights(g; w_max=second_max) for _ in 1:nb_scenarios]

    return TwoStageSpanningTreeInstance(
        g,
        edge_ind,
        nb_scenarios,
        first_stage_weights_matrix,
        first_stage_weights_vector,
        second_stage_weights,
    )
end

"""
    evaluate_first_stage_solution(inst::TwoStageSpanningTreeInstance, forest)

Returns the value of the solution of `inst` with `forest` as first stage solution.
"""
function evaluate_first_stage_solution(inst::TwoStageSpanningTreeInstance, forest)
    value = 0.0
    if length(forest) > 0
        value = sum(inst.first_stage_weights_matrix[src(e), dst(e)] for e in forest)
    end

    Threads.@threads for i in 1:(inst.nb_scenarios)
        m = minimum([inst.second_stage_weights[i][src(e), dst(e)] for e in edges(inst.g)])
        m = min(0, m - 1)
        weights = deepcopy(inst.second_stage_weights[i])
        for e in forest
            weights[src(e), dst(e)] = m
            weights[dst(e), src(e)] = m
        end
        tree_i = kruskal_mst(inst.g, weights)
        forest_i = setdiff(tree_i, forest)
        value_i = 0
        if length(forest_i) > 0
            value_i =
                1 / inst.nb_scenarios *
                sum(inst.second_stage_weights[i][src(e), dst(e)] for e in forest_i)
        end
        value += value_i
    end

    return value
end

"""
    kruskal_on_first_scenario_instance(instance::TwoStageSpanningTreeInstance)

Applies Kruskal algorithm with weight inst.first_stage_weights + inst.second_stage_weights[1]

Return value, first_stage_value, first_stage_solution
"""
function kruskal_on_first_scenario_instance(instance::TwoStageSpanningTreeInstance)
    best_weights = spzeros(nv(instance.g), nv(instance.g))
    for e in edges(instance.g)
        best_weights[src(e), dst(e)] = min(
            instance.first_stage_weights_matrix[src(e), dst(e)],
            instance.second_stage_weights[1][src(e), dst(e)],
        )
        best_weights[dst(e), src(e)] = min(
            instance.first_stage_weights_matrix[src(e), dst(e)],
            instance.second_stage_weights[1][src(e), dst(e)],
        )
    end
    value, tree = kruskal_mst_value(instance.g, best_weights)
    first_stage_solution = [
        e for e in tree if instance.first_stage_weights_matrix[src(e), dst(e)] <=
        instance.second_stage_weights[1][src(e), dst(e)]
    ]
    first_stage_value = sum(
        instance.first_stage_weights_matrix[src(e), dst(e)] for e in first_stage_solution
    )
    return value, first_stage_value, first_stage_solution
end
