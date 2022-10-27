using Flux
using Graphs
using GraphNeuralNetworks
using InferOpt
using JLD2
using ProgressMeter
using Random
using Statistics: mean
using MinimumWeightTwoStageSpanningTree
using UnicodePlots

include("utils.jl")

(; large_train_data, large_val_data, large_test_data) = build_or_load_datasets(;
    slice=1:12, reduce=true
);

gnn_large_train_data = build_gnn_dataset(large_train_data);
gnn_large_val_data = build_gnn_dataset(large_val_data);
gnn_large_test_data = build_gnn_dataset(large_test_data);

data_sets = [
    ("large train set", large_train_data),
    ("large validation set", large_val_data),
    ("large test set", large_test_data),
];

gnn_data_sets = [
    ("large train set", gnn_large_train_data),
    ("large validation set", gnn_large_val_data),
    ("large test set", gnn_large_test_data),
];

nb_features = size(large_train_data[1][1], 1)

function glm(; seed=1, nb_epochs=10, lr=0.01, name="glm")
    Random.seed!(seed)
    glm = Chain(Dense(nb_features => 1; bias=false), vec)
    trained_glm, losses, val_gaps = train_model(
        glm, shuffle(large_train_data), large_val_data; nb_epochs, seed, lr, name
    )
    evaluate_model(trained_glm, data_sets)
    return trained_glm, losses, val_gaps
end

function gnn(; seed=1, nb_epochs=10, lr=0.001, name="gnn")
    hidden_size = 50
    Random.seed!(seed)
    gnn_model = Chain(
        GNNChain(
            GraphConv(nb_features => hidden_size, relu),
            GraphConv(hidden_size => hidden_size, relu),
            GraphConv(hidden_size => hidden_size),
            Dense(hidden_size => 1; bias=false),
            vec,
        ),
        x -> x.ndata.x,
    )
    trained_gnn, losses, val_gaps = train_model(
        gnn_model,
        shuffle(gnn_large_train_data),
        gnn_large_val_data;
        nb_epochs,
        seed,
        lr,
        name,
    )
    evaluate_model(trained_gnn, gnn_data_sets)
    return trained_gnn, losses, val_gaps
end

glm_model, glm_losses, glm_val_gaps = glm(; seed=2, nb_epochs=20, name="glmfull20epochs")
println(lineplot(1:10, glm_val_gaps[2:end]))
glm_losses
glm_val_gaps

gnn_model, gnn_losses, gnn_val_gaps = gnn(; seed=2, nb_epochs=20, name="gnnfull20epochs")
println(lineplot(gnn_val_gaps))
println(lineplot(gnn_losses))
gnn_losses
gnn_val_gaps

data_glm = load("scripts/glmrestricted20epochs.jld2")
glm_model = data_glm["model"]
data_glm["epoch"]
evaluate_model(glm_model, data_sets)

data_gnn = load("scripts/gnnrestricted20epochs.jld2")
gnn_model = x -> data_gnn["model"](x).ndata.x
data_gnn["epoch"]
evaluate_model(gnn_model, gnn_data_sets)
