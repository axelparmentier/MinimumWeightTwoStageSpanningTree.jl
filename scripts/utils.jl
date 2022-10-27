function compute_σ(X)
    nb_arcs = 0
    nb_features = size(X[1][1], 1)
    μ = zeros(nb_features)
    σ = zeros(nb_features)

    for (x, _, (_, _, _, _)) in X
        nb_edges = size(x, 2)
        μ .+= sum(x; dims=2)
        nb_arcs += nb_edges
    end
    μ ./= nb_arcs

    for (x, _, (_, _, _, _)) in X
        σ .+= sum((x .- μ) .^ 2; dims=2)
    end
    σ ./= nb_arcs
    σ = sqrt.(σ)
    return σ
end

function reduce_data!(X, σ)
    for (x, _, (_, _, _, _)) in X
        for col in axes(x, 2) #  1:size(x, 2)
            @. x[:, col] = @views (x[:, col]) / σ
        end
    end
end

function build_dataset(;
    nb_scenarios=5:5:20,
    first_max=20:20,
    second_max=10:5:30,
    seeds=1:5,
    grid_sizes=4:6,
    solver=benders_solver,
    slice=nothing,
)
    train_set_params = []
    for gs in grid_sizes
        for ns in nb_scenarios
            for fm in first_max
                for sm in second_max
                    for seed in seeds
                        push!(train_set_params, (gs, ns, fm, sm, seed))
                    end
                end
            end
        end
    end

    res = [
        build_solve_and_encode_instance_as_maximum_weight_forest(;
            grid_size=gs,
            seed=seed,
            nb_scenarios=ns,
            first_max=fm,
            second_max=sm,
            solver=solver,
        ) for (gs, ns, fm, sm, seed) in train_set_params
    ]
    if isnothing(slice)
        return [(x, y, (inst, lb, ub, sol)) for (x, y, (inst, lb, ub, sol)) in res]
    end
    # else
    return [(x[slice, :], y, (inst, lb, ub, sol)) for (x, y, (inst, lb, ub, sol)) in res]
end

"""
    build_or_load_datasets(; slice=nothing, reduce=true)

Arguments:
- `slice`: if is not nothing, slice the features
- `reduce`: reduce data by std of large training data if true
"""
function build_or_load_datasets(; slice=nothing, reduce::Bool=true)
    nb_scenarios = 5:5:20
    first_max = 20:20
    second_max = 10:5:30
    train_seeds = 1:5
    val_seeds = 51:55
    test_seeds = 101:105
    #small_sizes = 4:6
    large_sizes = 10:10:60

    # @info 1

    # train_data = build_dataset(;
    #     nb_scenarios=nb_scenarios,
    #     first_max=first_max,
    #     second_max=second_max,
    #     seeds=train_seeds,
    #     grid_sizes=small_sizes,
    #     solver=TwoStageSpanningTree.benders_solver,
    #     slice,
    # )

    # @info 2

    # val_data = build_dataset(;
    #     nb_scenarios=nb_scenarios,
    #     first_max=first_max,
    #     second_max=second_max,
    #     seeds=val_seeds,
    #     grid_sizes=small_sizes,
    #     solver=TwoStageSpanningTree.benders_solver,
    #     slice,
    # )

    # @info 3

    # test_data = build_dataset(;
    #     nb_scenarios=nb_scenarios,
    #     first_max=first_max,
    #     second_max=second_max,
    #     seeds=test_seeds,
    #     grid_sizes=small_sizes,
    #     solver=TwoStageSpanningTree.benders_solver,
    #     slice,
    # )

    # @info 4

    # lagrangian_test_data = build_dataset(;
    #     nb_scenarios=nb_scenarios,
    #     first_max=first_max,
    #     second_max=second_max,
    #     seeds=test_seeds,
    #     grid_sizes=small_sizes,
    #     solver=lagrangian_heuristic_solver,
    #     slice,
    # )

    @info 5

    large_train_data = build_dataset(;
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        seeds=train_seeds,
        grid_sizes=large_sizes,
        solver=lagrangian_heuristic_solver,
        slice,
    )

    @info 6

    seeds = 51:55
    large_val_data = build_dataset(;
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        seeds=val_seeds,
        grid_sizes=large_sizes,
        solver=lagrangian_heuristic_solver,
        slice,
    )

    @info 7

    seeds = 56:60
    large_test_data = build_dataset(;
        nb_scenarios=nb_scenarios,
        first_max=first_max,
        second_max=second_max,
        seeds=test_seeds,
        grid_sizes=large_sizes,
        solver=lagrangian_heuristic_solver,
        slice,
    )

    @info 8

    #@info "small" compute_σ(train_data)
    #@info "large" compute_σ(large_train_data)
    if reduce
        σ = compute_σ(large_train_data)
        for dataset in [large_train_data, large_val_data, large_test_data]
            reduce_data!(dataset, σ)
        end
    end

    return (; large_train_data, large_val_data, large_test_data)
end

function compute_edge_graph(inst, x)
    nb_edges = ne(inst.g)
    adj = falses(nb_edges, nb_edges)
    for edge in edges(inst.g)
        u1, u2 = src(edge), dst(edge)
        idx = inst.edge_index[u1, u2]
        for u in (u1, u2)
            for v in neighbors(inst.g, u)
                idx2 = inst.edge_index[u, v]
                if idx == idx2
                    continue
                end
                adj[idx, idx2] = 1
            end
        end
    end
    edge_graph = SimpleGraph(adj)
    egraph = GNNGraph(edge_graph; ndata=x)
    return egraph
end

function build_gnn_dataset(dataset)
    return [
        (compute_edge_graph(inst, x), y, (inst, lb, ub, sol)) for
        (x, y, (inst, lb, ub, sol)) in dataset
    ]
end

function train_setup(initial_model, ε, nb_samples, seed, lr)
    Random.seed!(seed)
    perturbed_maximizer = PerturbedAdditive(
        maximum_weight_forest_linear_maximizer; ε, nb_samples
    )
    loss = FenchelYoungLoss(perturbed_maximizer)

    model = deepcopy(initial_model)
    opt = Adam(lr)
    return (; loss, model, opt)
end

function eval_gap(model, data)
    return mean(
        100 * (
            evaluate_first_stage_solution(
                inst, kruskal_maximum_weight_forest(model(x), inst)
            ) - lb
        ) / lb for (x, y, (inst, lb, ub, sol)) in data
    )
end

function train_model(
    initial_model,
    train_data,
    val_data;
    nb_epochs=200,
    ε=1.0,
    nb_samples=20,
    seed=0,
    lr=0.001,
    eval_every=1,
    name="glm",
)
    (; loss, model, opt) = train_setup(initial_model, ε, nb_samples, seed, lr)
    losses = Float64[]
    val_gaps = Float64[eval_gap(model, val_data)]
    best_gap = Inf
    @showprogress for epoch in 1:nb_epochs
        l = 0.0
        for (x, y, (inst, _, _, _)) in train_data
            grads = gradient(Flux.params(model)) do
                l += loss(model(x), y; inst=inst)
            end
            Flux.update!(opt, Flux.params(model), grads)
        end
        push!(losses, l)
        if epoch % eval_every == 0
            val_gap = eval_gap(model, val_data)
            push!(val_gaps, val_gap)
            if val_gap < best_gap
                jldsave("scripts/$name.jld2"; model=model, epoch=epoch)
                best_gap = val_gap
            end
        end
    end
    println(lineplot(losses; xlabel="Epoch", ylabel="Loss"))
    return model, losses, val_gaps
end

function evaluate_model(model, datasets)
    pipeline(x, inst) = kruskal_maximum_weight_forest(model(x), inst)
    for (data_name, data_set) in datasets
        gaps_lb = Float64[]
        gaps_ub = Float64[]
        for (x, y, (inst, lb, ub, sol)) in data_set
            gap_lb =
                100 * (evaluate_first_stage_solution(inst, pipeline(x, inst)) - lb) / lb
            push!(gaps_lb, gap_lb)
            gap_ub =
                100 * (evaluate_first_stage_solution(inst, pipeline(x, inst)) - ub) / ub
            push!(gaps_ub, gap_ub)
        end
        # println(histogram(gaps_lb; nbins=10, name=" " * data_name * " gap lb"))
        # println(histogram(gaps_ub; nbins=10, name=" " * data_name * " gap ub"))
        println("---")
        average_gap_lb = round(sum(gaps_lb) / length(gaps_lb); digits=2)
        min_gap_lb = round(minimum(gaps_lb); digits=2)
        max_gap_lb = round(maximum(gaps_lb); digits=2)
        min_gap_ub = round(minimum(gaps_ub); digits=2)
        max_gap_ub = round(maximum(gaps_ub); digits=2)
        average_gap_ub = round(sum(gaps_ub) / length(gaps_ub); digits=2)
        @info "$data_name" average_gap_lb min_gap_lb max_gap_lb average_gap_ub min_gap_ub max_gap_ub
    end
end
